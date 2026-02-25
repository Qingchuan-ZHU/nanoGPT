param(
  [int]$Port = 8000,
  [string]$Page = "index.html",
  [switch]$Restart,
  [switch]$ReleaseOnly
)

$ErrorActionPreference = "Stop"

$studyDir = $PSScriptRoot
$pagePath = Join-Path $studyDir $Page
$pidFile = Join-Path $studyDir ".open-study-$Port.pid"

if (-not (Test-Path -Path $pagePath -PathType Leaf)) {
  throw "Page not found: $Page"
}

$url = "http://localhost:$Port/$Page"

function Get-ListeningProcessIds {
  param([int]$LocalPort)

  try {
    return @(
      Get-NetTCPConnection -State Listen -LocalPort $LocalPort -ErrorAction Stop |
        Select-Object -ExpandProperty OwningProcess -Unique
    )
  } catch {
    return @()
  }
}

function Stop-ListenersOnPort {
  param([int]$LocalPort)

  $pids = Get-ListeningProcessIds -LocalPort $LocalPort | Where-Object { $_ -ne $PID }
  if (-not $pids.Count) {
    return
  }

  foreach ($procId in $pids) {
    $proc = Get-Process -Id $procId -ErrorAction SilentlyContinue
    $procName = if ($proc) { $proc.ProcessName } else { "UnknownProcess" }
    Write-Host "Freeing port ${LocalPort}: stopping $procName ($procId)"
    Stop-Process -Id $procId -Force -ErrorAction Stop
  }

  Start-Sleep -Milliseconds 300
  $remaining = Get-ListeningProcessIds -LocalPort $LocalPort | Where-Object { $_ -ne $PID }
  if ($remaining.Count) {
    throw "Port $LocalPort is still in use: $($remaining -join ', ')"
  }
}

function Stop-ExistingServerInstance {
  param([string]$PidFilePath)

  if (-not (Test-Path -Path $PidFilePath -PathType Leaf)) {
    return
  }

  $raw = (Get-Content -Path $PidFilePath -ErrorAction SilentlyContinue | Select-Object -First 1)
  $savedPid = 0
  if (-not [int]::TryParse([string]$raw, [ref]$savedPid)) {
    Remove-Item -Path $PidFilePath -Force -ErrorAction SilentlyContinue
    return
  }

  if ($savedPid -eq $PID) {
    return
  }

  $proc = Get-Process -Id $savedPid -ErrorAction SilentlyContinue
  if ($proc) {
    Write-Host "Stopping previous study server instance: $savedPid"
    Stop-Process -Id $savedPid -Force -ErrorAction SilentlyContinue
    Start-Sleep -Milliseconds 250
  }

  Remove-Item -Path $PidFilePath -Force -ErrorAction SilentlyContinue
}

function Get-ContentType {
  param([string]$Path)

  $ext = ([System.IO.Path]::GetExtension($Path)).ToLowerInvariant()
  switch ($ext) {
    ".html" { return "text/html; charset=utf-8" }
    ".css" { return "text/css; charset=utf-8" }
    ".js" { return "application/javascript; charset=utf-8" }
    ".json" { return "application/json; charset=utf-8" }
    ".png" { return "image/png" }
    ".jpg" { return "image/jpeg" }
    ".jpeg" { return "image/jpeg" }
    ".gif" { return "image/gif" }
    ".svg" { return "image/svg+xml" }
    ".ico" { return "image/x-icon" }
    ".txt" { return "text/plain; charset=utf-8" }
    default { return "application/octet-stream" }
  }
}

function Write-ErrorResponse {
  param(
    [System.Net.HttpListenerResponse]$Response,
    [int]$StatusCode,
    [string]$Message
  )

  $bytes = [System.Text.Encoding]::UTF8.GetBytes($Message)
  $Response.StatusCode = $StatusCode
  $Response.ContentType = "text/plain; charset=utf-8"
  $Response.ContentLength64 = $bytes.Length
  $Response.OutputStream.Write($bytes, 0, $bytes.Length)
  $Response.OutputStream.Close()
}

function Start-ListenerWithRetry {
  param(
    [int]$LocalPort,
    [int]$MaxAttempts = 10,
    [int]$DelayMs = 250
  )

  $lastError = $null
  for ($i = 1; $i -le $MaxAttempts; $i++) {
    $candidate = [System.Net.HttpListener]::new()
    $candidate.Prefixes.Add("http://localhost:$LocalPort/")

    try {
      $candidate.Start()
      if ($i -gt 1) {
        Write-Host "Listener started on attempt $i."
      }
      return $candidate
    } catch {
      $lastError = $_
      try { $candidate.Close() } catch {}
      Stop-ListenersOnPort -LocalPort $LocalPort
      Start-Sleep -Milliseconds $DelayMs
    }
  }

  throw "Failed to start listener on port $LocalPort after $MaxAttempts attempts. Last error: $($lastError.Exception.Message)"
}

if ($Restart) {
  Stop-ExistingServerInstance -PidFilePath $pidFile
}

Stop-ListenersOnPort -LocalPort $Port

if ($ReleaseOnly) {
  Write-Host "Port $Port released. Exit because -ReleaseOnly was specified."
  exit 0
}

$listener = $null

try {
  $listener = Start-ListenerWithRetry -LocalPort $Port
  Set-Content -Path $pidFile -Value "$PID" -NoNewline
  Start-Process $url | Out-Null

  Write-Host "Study page opened: $url"
  Write-Host "Server PID: $PID"
  Write-Host "Tip: use -Restart to stop old instance and relaunch."
  Write-Host "Keep this window open. Ctrl+C or closing this window will release the port."

  $baseFullPath = [System.IO.Path]::GetFullPath($studyDir)

  while ($listener.IsListening) {
    try {
      $context = $listener.GetContext()
    } catch [System.Net.HttpListenerException] {
      break
    }

    $request = $context.Request
    $response = $context.Response

    try {
      $relative = [System.Uri]::UnescapeDataString($request.Url.AbsolutePath.TrimStart("/"))
      if ([string]::IsNullOrWhiteSpace($relative)) {
        $relative = $Page
      }

      $targetPath = Join-Path $studyDir $relative
      $targetFullPath = [System.IO.Path]::GetFullPath($targetPath)

      if (-not $targetFullPath.StartsWith($baseFullPath, [System.StringComparison]::OrdinalIgnoreCase)) {
        Write-ErrorResponse -Response $response -StatusCode 403 -Message "403 Forbidden"
        continue
      }

      if (Test-Path -Path $targetFullPath -PathType Container) {
        $targetFullPath = Join-Path $targetFullPath "index.html"
      }

      if (-not (Test-Path -Path $targetFullPath -PathType Leaf)) {
        Write-ErrorResponse -Response $response -StatusCode 404 -Message "404 Not Found"
        continue
      }

      $bytes = [System.IO.File]::ReadAllBytes($targetFullPath)
      $response.StatusCode = 200
      $response.ContentType = Get-ContentType -Path $targetFullPath
      $response.ContentLength64 = $bytes.Length
      $response.OutputStream.Write($bytes, 0, $bytes.Length)
      $response.OutputStream.Close()
    } catch {
      if ($response.OutputStream.CanWrite) {
        Write-ErrorResponse -Response $response -StatusCode 500 -Message "500 Internal Server Error"
      }
    }
  }
} finally {
  if (Test-Path -Path $pidFile -PathType Leaf) {
    $raw = (Get-Content -Path $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1)
    $savedPid = 0
    if ([int]::TryParse([string]$raw, [ref]$savedPid) -and $savedPid -eq $PID) {
      Remove-Item -Path $pidFile -Force -ErrorAction SilentlyContinue
    }
  }

  if ($listener) {
    try { $listener.Stop() } catch {}
    try { $listener.Close() } catch {}
  }

  $left = Get-ListeningProcessIds -LocalPort $Port | Where-Object { $_ -eq $PID }
  if ($left.Count) {
    Write-Host "Port $Port is still owned by this process."
  } else {
    Write-Host "Port $Port released."
  }
}
