# Fraud Detection Project
library(rpart)
library(rpart.plot)
library(rsample)
library(caret)
library(klaR)

data <- read.csv("alzheimers_disease_data.csv")

#1. Describe data through Visuals (Charts/Tables/Graphs, etc)
head(data)
summary(data)
dim(data)

str(data)
# Converting each feature into a factor with their respective labels
# Demographic detail factors
data$Gender <- factor(data$Gender, levels = c(0, 1), labels=c("Male", "Female"))
data$Ethnicity <- factor(data$Ethnicity, levels = c(0, 1, 2, 3), labels=c("Caucasian", "African American", "Asian", "Other"))
data$EducationLevel <- factor(data$EducationLevel, levels = c(0, 1, 2, 3), 
                              labels = c("None", "High School", "Bachelor's", "Higher"))

# Binary lifestyle factors
data$Smoking <- factor(data$Smoking, levels = c(0, 1), labels = c("No", "Yes"))
data$FamilyHistoryAlzheimers <- factor(data$FamilyHistoryAlzheimers, levels = c(0, 1), labels = c("No", "Yes"))
data$CardiovascularDisease <- factor(data$CardiovascularDisease, levels = c(0, 1), labels = c("No", "Yes"))
data$Diabetes <- factor(data$Diabetes, levels = c(0, 1), labels = c("No", "Yes"))
data$Depression <- factor(data$Depression, levels = c(0, 1), labels = c("No", "Yes"))
data$HeadInjury <- factor(data$HeadInjury, levels = c(0, 1), labels = c("No", "Yes"))
data$Hypertension <- factor(data$Hypertension, levels = c(0, 1), labels = c("No", "Yes"))
data$MemoryComplaints <- factor(data$MemoryComplaints, levels = c(0, 1), labels = c("No", "Yes"))
data$BehavioralProblems <- factor(data$BehavioralProblems, levels = c(0, 1), labels = c("No", "Yes"))

# Binary symptom indicators
data$Confusion <- factor(data$Confusion, levels = c(0, 1), labels = c("No", "Yes"))
data$Disorientation <- factor(data$Disorientation, levels = c(0, 1), labels = c("No", "Yes"))
data$PersonalityChanges <- factor(data$PersonalityChanges, levels = c(0, 1), labels = c("No", "Yes"))
data$DifficultyCompletingTasks <- factor(data$DifficultyCompletingTasks, levels = c(0, 1), labels = c("No", "Yes"))
data$Forgetfulness <- factor(data$Forgetfulness, levels = c(0, 1), labels = c("No", "Yes"))

# Diagnosis info
data$Diagnosis <- factor(data$Diagnosis, levels = c(0, 1), labels = c("No", "Yes"))

head(data)
diagnosis <- data$Diagnosis
tableDiagnosis <- table(diagnosis)
proportionDiagnosis <- prop.table(tableDiagnosis)

# Diagnosis of Alzheimer's Disease Distribution in data set
barplot(tableDiagnosis, col="lightblue", main="Distribution of Diagnosis", xlab="Diagnosis", ylab="# of Observations")

age <- data$Age
tableAge <- table(age)
barplot(tableAge, col="khaki", main="Distribution of Ages", xlab="Age", ylab="# of Observations")

plot(diagnosis, data$Age, col="coral", xlab="Diagnosis", ylab="Age", main="Diagnosis vs Age")



#2. Classification Pr#2. Classification Pr#2. Classification Procedures
  # Rpart Model
    # Show Confusion Matrix from Decision Tree
  # Naive Bayes Model
    # Use classification variable and predictors chosen from rpart
    # Show Confusion Matrix
  # Compare accuracies