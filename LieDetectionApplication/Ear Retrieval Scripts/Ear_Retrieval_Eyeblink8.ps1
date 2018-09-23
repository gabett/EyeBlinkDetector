
function Main([string] $FilePath){

    if($FilePath.Length -le 0){
        Write-Host "Invalid path." -ForegroundColor Red
        return;
    }

    EarRetrieval($FilePath)

}

function EarRetrieval([string] $TagFilePath)
{
    $outputFilePath = "D:\COB\Eyeblink8\11\EarIntervals.txt"

    if(Test-Path $outputFilePath)
    {
      Remove-Item -Path $outputFilePath
    }
    
    New-Item $outputFilePath -ItemType File

    foreach($line in Get-Content $TagFilePath){
        $rowValues = $line.Split(" ")

        if ($rowValues[1].ToLower() -like "*close*"){

            [string] $leftBoundary = [int]$rowValues[0] - 6
            [string] $rightBoundary = [int]$rowValues[0] + 6

            $formattedOutput = $leftBoundary + "    " + $rightBoundary.ToString()

            $formattedOutput | Add-Content -Path $outputFilePath
        }
    }
}

Clear-Host
Main -FilePath "D:\COB\Eyeblink8\11\27122013_154548_cam.tag"