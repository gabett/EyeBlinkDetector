#requires -version 2
<#
.SYNOPSIS
  EAR calculation and formatting for 300VW_Dataset_2015_12_14.
.DESCRIPTION
    For each video of 300VW_Dataset_2015_12_14, the script will format the landmark of each frame in a more handable way
  and computes the EAR. At the end of the video analysis, a .txt file is created containing all the landmarks, sorted by
  Eeye Aspect Ratio, descending.
.INPUTS
  Path: string containing the path to 300VW_Dataset_2015_12_14 folder.
.OUTPUTS
  A .txt file called folderVideoName.txt containing all the landmarks with their corresponding EAR, sorted descending.
.NOTES
  Version:        1.0
  Author:         Gabriele Etta and Rocco Cattani 
  Creation Date:  1/04/2018
  Purpose/Change: Initial script development
  
.EXAMPLE
  <Example goes here. Repeat this attribute for more than one example>
#>

#----------------------------------------------------------[Declarations]----------------------------------------------------------

$defaultPath = "D:\Dati progetto COB\300VW_Dataset_2015_12_14"


#-----------------------------------------------------------[Functions]------------------------------------------------------------

function Main ([string]$Path) {
  if ([string]::IsNullOrEmpty($Path)){
    $FolderPath = $defaultPath
  }
  else 
  {
    $FolderPath = $Path
  }
  ReadLandmarks($FolderPath)
}

function ReadLandmarks([string]$Path) {

 $folders = Get-ChildItem $Path | Select-Object -ExpandProperty FullName
 $trainingSet = $Path + "\TrainingSet.txt"
 $landmarksEarDict = @{}
 $openLandmarksAndEar = @()
 $closedLandmarksAndEar = @()

 if(Test-Path $trainingSet)
 {
   Remove-Item -Path $trainingSet
 }

 foreach($folder in $folders)
 {
   Write-Host "Video Number $folder" -ForegroundColor Green  
   $landmarkFolder = Get-ChildItem $folder

   foreach($annotFolder in $landmarkFolder[0] | Select-Object -ExpandProperty FullName)
   {
      foreach($ptsFile in Get-ChildItem $annotFolder)
      {
        $fileContent = Get-Content $ptsFile.FullName

        $splittedFile = $fileContent.Split("\t  ")

        $landmarkArray = $splittedFile[6..73]

        # Computing euclidean distances

        $middleLeftLeftEyeCoordinates = $landmarkArray[36].Split("  ")
        $upperLeftLeftEyeCoordinates = $landmarkArray[37].Split("  ") 
        $upperRightLeftEyeCoordinates = $landmarkArray[38].Split("  ")
        $middleRightLeftEyeCoordinates = $landmarkArray[39].Split("  ")
        $underRightLeftEyeCoordinates = $landmarkArray[40].Split("  ")
        $underLeftLeftEyeCoordinates = $landmarkArray[41].Split("  ") 

        $verticalLeftLandmarksEuclidean =  
          [math]::Sqrt([math]::Pow(($upperLeftLeftEyeCoordinates[0] - $underLeftLeftEyeCoordinates[0]),2) + [math]::Pow(($upperLeftLeftEyeCoordinates[1] - $underLeftLeftEyeCoordinates[1]),2))

        $verticalRightLandmarksEuclidean =  
          [math]::sqrt([math]::Pow(($upperRightLeftEyeCoordinates[0] - $underRightLeftEyeCoordinates[0]),2) + [math]::Pow(($upperRightLeftEyeCoordinates[1] - $underRightLeftEyeCoordinates[1]),2))

        $horizontalLandmarksEuclidean =  
          [math]::Sqrt([math]::Pow(($middleLeftLeftEyeCoordinates[0] - $middleRightLeftEyeCoordinates[0]),2) + [math]::Pow(($middleLeftLeftEyeCoordinates[1] - $middleRightLeftEyeCoordinates[1]),2))
       
        $EAR = ($verticalLeftLandmarksEuclidean + $verticalRightLandmarksEuclidean) / (2*($horizontalLandmarksEuclidean))

        $landmarksEarDict.Add($landmarkArray, $EAR)
      }
      
      $landmarksOrderedByEar = $landmarksEarDict.GetEnumerator() | Sort-Object Value -Descending

      $openEyes = $landmarksOrderedByEar | Select-Object -First 10 
      foreach($value in $openEyes)
      {
        $openLandmarksAndEar += ,@($value.Key, $value.Value, "1") # 1 Means opened eyes
      }
      $openLandmarksAndEar |  Add-Content -Path $trainingSet
      
      $closedEyes = $landmarksOrderedByEar | Select-Object -Last 10
      foreach($value in $closedEyes)
      {
        $closedLandmarksAndEar += ,@($value.Key, $value.Value, "0") # 0 Means closed eyes
      }
      $closedLandmarksAndEar |  Add-Content -Path $trainingSet
   }
 }
}

#-----------------------------------------------------------[Execution]------------------------------------------------------------

Clear-Host
Main -Path "C:\Users\gabri\Desktop\300VW_Dataset_2015_12_14"