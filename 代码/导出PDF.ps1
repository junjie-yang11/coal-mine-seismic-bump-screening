param([Parameter(Mandatory=$true)][string]$DocumentPath)
$ErrorActionPreference = 'Stop'
$sourcePath = (Resolve-Path -LiteralPath $DocumentPath).Path
$pdfPath = [System.IO.Path]::ChangeExtension($sourcePath, '.pdf')
$wordApp = $null
$wordDoc = $null
try {
    $wordApp = New-Object -ComObject Word.Application
    $wordApp.Visible = $false
    $wordApp.DisplayAlerts = 0
    $wordDoc = $wordApp.Documents.Open($sourcePath, $false, $true)
    $wordDoc.Repaginate()
    $wordDoc.ExportAsFixedFormat($pdfPath, 17, $false, 0, 0, 1, 1, 0, $false)
    Write-Output $pdfPath
    Write-Output ('Pages: ' + $wordDoc.ComputeStatistics(2))
} finally {
    if ($wordDoc) {
        $wordDoc.Close(0)
        [Runtime.InteropServices.Marshal]::ReleaseComObject($wordDoc) | Out-Null
    }
    if ($wordApp) {
        $wordApp.Quit(0)
        [Runtime.InteropServices.Marshal]::ReleaseComObject($wordApp) | Out-Null
    }
}
