#Set up the working directory path
direc <- "C:\\Users\\prate\\Desktop\\ICT_solution\\Data\\unmerged_data\\clickbait"
setwd(direc)
#Get list of all the files in the given directory 
CBfiles <- list.files()

#Creating a data frame for clickbait videos
finalCB <- data.frame(Date=as.Date(character()),
                      File=character(), 
                      User=character(), 
                      stringsAsFactors=FALSE)

#Reading the first csv file
finalCB <- read.csv("channel1.csv", header=T, na.strings=c(""), stringsAsFactors = T)
#Converting it to data frame
finalCB <- as.data.frame(finalCB)
#assigning tag of 1 to the clickbait videos
finalCB$tag <- 1
#Setting up a variable initialised to 0
i = 0
#FOR loop for iterating through all the files in the given directory 
for (CBfile in CBfiles) {
  if(i == 0){
    cat(i)
  }
  i = i + 1
  #Reading all the values of the given csv file
  CBData <- read.csv(CBfile, header=T, na.strings=c(""), stringsAsFactors = T)
  #Converting it to a data frame
  CBData <- as.data.frame(CBData)
  #Merging the data to the finalCB data frame
  if(dim(CBData)[1] != 0){
    CBData$tag <- 1
    finalCB <- rbind(finalCB,CBData)
  }
}
#Setting up the new directory path
direc <- "C:\\Users\\prate\\Desktop\\ICT_solution\\Data\\unmerged_data\\nonClickbait\\Animals"
setwd(direc)
NCBfiles <- list.files()

finalNCB <- data.frame(Date=as.Date(character()),
                       File=character(), 
                       User=character(), 
                       stringsAsFactors=FALSE)

finalNCB <- read.csv("channel1.csv", header=T, na.strings=c(""), stringsAsFactors = T)
finalNCB <- as.data.frame(finalNCB)
finalNCB$tag <- 0
j = 0
for (NCBfile in NCBfiles) {
  if(j == 0){
    cat(j)
  }
  j = j + 1
  NCBData <- read.csv(NCBfile, header=T, na.strings=c(""), stringsAsFactors = T)
  NCBData <- as.data.frame(NCBData)
  if(dim(NCBData)[1] != 0){
    NCBData$tag <- 0
    finalNCB <- rbind(finalNCB,NCBData)
  }
}

#Setting up the new directory path
direc <- "C:\\Users\\prate\\Desktop\\ICT_solution\\Data\\unmerged_data\\nonClickbait\\Animals"
setwd(direc)
NCBfiles <- list.files()
for (NCBfile in NCBfiles) {
  NCBData <- read.csv(NCBfile, header=T, na.strings=c(""), stringsAsFactors = T)
  NCBData <- as.data.frame(NCBData)
  if(dim(NCBData)[1] != 0){
    NCBData$tag <- 0
    finalNCB <- rbind(finalNCB,NCBData)
  }
}

#Setting up the new directory path
direc <- "C:\\Users\\prate\\Desktop\\ICT_solution\\Data\\unmerged_data\\nonClickbait\\Comedy"
setwd(direc)
NCBfiles <- list.files()
for (NCBfile in NCBfiles) {
  NCBData <- read.csv(NCBfile, header=T, na.strings=c(""), stringsAsFactors = T)
  NCBData <- as.data.frame(NCBData)
  if(dim(NCBData)[1] != 0){
    NCBData$tag <- 0
    finalNCB <- rbind(finalNCB,NCBData)
  }
}

#Setting up the new directory path
direc <- "C:\\Users\\prate\\Desktop\\ICT_solution\\Data\\unmerged_data\\nonClickbait\\Education"
setwd(direc)
NCBfiles <- list.files()
for (NCBfile in NCBfiles) {
  NCBData <- read.csv(NCBfile, header=T, na.strings=c(""), stringsAsFactors = T)
  NCBData <- as.data.frame(NCBData)
  if(dim(NCBData)[1] != 0){
    NCBData$tag <- 0
    finalNCB <- rbind(finalNCB,NCBData)
  }
}

#Setting up the new directory path
direc <- "C:\\Users\\prate\\Desktop\\ICT_solution\\Data\\unmerged_data\\nonClickbait\\Family"
setwd(direc)
NCBfiles <- list.files()
for (NCBfile in NCBfiles) {
  NCBData <- read.csv(NCBfile, header=T, na.strings=c(""), stringsAsFactors = T)
  NCBData <- as.data.frame(NCBData)
  if(dim(NCBData)[1] != 0){
    NCBData$tag <- 0
    finalNCB <- rbind(finalNCB,NCBData)
  }
}

#Setting up the new directory path
direc <- "C:\\Users\\prate\\Desktop\\ICT_solution\\Data\\unmerged_data\\nonClickbait\\Gaming"
setwd(direc)
NCBfiles <- list.files()
for (NCBfile in NCBfiles) {
  NCBData <- read.csv(NCBfile, header=T, na.strings=c(""), stringsAsFactors = T)
  NCBData <- as.data.frame(NCBData)
  if(dim(NCBData)[1] != 0){
    NCBData$tag <- 0
    finalNCB <- rbind(finalNCB,NCBData)
  }
}

#Setting up the new directory path
direc <- "C:\\Users\\prate\\Desktop\\ICT_solution\\Data\\unmerged_data\\nonClickbait\\Movies"
setwd(direc)
NCBfiles <- list.files()
for (NCBfile in NCBfiles) {
  NCBData <- read.csv(NCBfile, header=T, na.strings=c(""), stringsAsFactors = T)
  NCBData <- as.data.frame(NCBData)
  if(dim(NCBData)[1] != 0){
    NCBData$tag <- 0
    finalNCB <- rbind(finalNCB,NCBData)
  }
}

#Setting up the new directory path
direc <- "C:\\Users\\prate\\Desktop\\ICT_solution\\Data\\unmerged_data\\nonClickbait\\Music"
setwd(direc)
NCBfiles <- list.files()
for (NCBfile in NCBfiles) {
  NCBData <- read.csv(NCBfile, header=T, na.strings=c(""), stringsAsFactors = T)
  NCBData <- as.data.frame(NCBData)
  if(dim(NCBData)[1] != 0){
    NCBData$tag <- 0
    finalNCB <- rbind(finalNCB,NCBData)
  }
}

#Setting up the new directory path
direc <- "C:\\Users\\prate\\Desktop\\ICT_solution\\Data\\unmerged_data\\nonClickbait\\News"
setwd(direc)
NCBfiles <- list.files()
for (NCBfile in NCBfiles) {
  NCBData <- read.csv(NCBfile, header=T, na.strings=c(""), stringsAsFactors = T)
  NCBData <- as.data.frame(NCBData)
  if(dim(NCBData)[1] != 0){
    NCBData$tag <- 0
    finalNCB <- rbind(finalNCB,NCBData)
  }
}

#Setting up the new directory path
direc <- "C:\\Users\\prate\\Desktop\\ICT_solution\\Data\\unmerged_data\\nonClickbait\\Science"
setwd(direc)
NCBfiles <- list.files()
for (NCBfile in NCBfiles) {
  NCBData <- read.csv(NCBfile, header=T, na.strings=c(""), stringsAsFactors = T)
  NCBData <- as.data.frame(NCBData)
  if(dim(NCBData)[1] != 0){
    NCBData$tag <- 0
    finalNCB <- rbind(finalNCB,NCBData)
  }
}

#Setting up the new directory path
direc <- "C:\\Users\\prate\\Desktop\\ICT_solution\\Data\\unmerged_data\\nonClickbait\\Sports"
setwd(direc)
NCBfiles <- list.files()
for (NCBfile in NCBfiles) {
  NCBData <- read.csv(NCBfile, header=T, na.strings=c(""), stringsAsFactors = T)
  NCBData <- as.data.frame(NCBData)
  if(dim(NCBData)[1] != 0){
    NCBData$tag <- 0
    finalNCB <- rbind(finalNCB,NCBData)
  }
}

#Setting up the new directory path
direc <- "C:\\Users\\prate\\Desktop\\ICT_solution\\Data\\unmerged_data\\nonClickbait\\Trailers"
setwd(direc)
NCBfiles <- list.files()
for (NCBfile in NCBfiles) {
  NCBData <- read.csv(NCBfile, header=T, na.strings=c(""), stringsAsFactors = T)
  NCBData <- as.data.frame(NCBData)
  if(dim(NCBData)[1] != 0){
    NCBData$tag <- 0
    finalNCB <- rbind(finalNCB,NCBData)
  }
}

#Setting up the new directory path
direc <- "C:\\Users\\prate\\Desktop\\ICT_solution\\Data\\unmerged_data\\nonClickbait\\Travel"
setwd(direc)
NCBfiles <- list.files()
for (NCBfile in NCBfiles) {
  NCBData <- read.csv(NCBfile, header=T, na.strings=c(""), stringsAsFactors = T)
  NCBData <- as.data.frame(NCBData)
  if(dim(NCBData)[1] != 0){
    NCBData$tag <- 0
    finalNCB <- rbind(finalNCB,NCBData)
  }
}

#Setting up the new directory path
direc <- "C:\\Users\\prate\\Desktop\\ICT_solution\\Data\\unmerged_data\\nonClickbait\\Vehicles"
setwd(direc)
NCBfiles <- list.files()
for (NCBfile in NCBfiles) {
  NCBData <- read.csv(NCBfile, header=T, na.strings=c(""), stringsAsFactors = T)
  NCBData <- as.data.frame(NCBData)
  if(dim(NCBData)[1] != 0){
    NCBData$tag <- 0
    finalNCB <- rbind(finalNCB,NCBData)
  }
}
#Setting up the new directory path where the final dataset needs to be stored
direc <- "C:\\Users\\prate\\Desktop\\ICT_solution\\Data\\final_data"
setwd(direc)

#Combining clickbait and non-clickbait data
finalData <- rbind(finalCB,finalNCB) 
#Feature Selection i.e. Selecting the data required for performing this research
finalData <- finalData[,c("video_title","tag")]
#Saving the data frame in CSV format
write.csv(finalData, file = "final1_try_channel.csv", row.names = FALSE)
