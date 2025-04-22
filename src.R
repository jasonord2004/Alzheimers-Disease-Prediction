# Fraud Detection Project
library(rpart)
library(rpart.plot)
library(rsample)
library(caret)
library(klaR)

set.seed(12345)
data <- read.csv("alzheimers_disease_data.csv")

#1. Describe data through Visuals (Charts/Tables/Graphs, etc)
head(data)
summary(data)
dim(data)

str(data)

#####
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

# Remove DoctorInCharge Column (All values are "XXXConfid")
data <- data[, -35]

##### 

# Data Visualization
head(data)
diagnosis <- data$Diagnosis
tableDiagnosis <- table(diagnosis)
proportionDiagnosis <- prop.table(tableDiagnosis)

# Diagnosis of Alzheimer's Disease Distribution in data set
barplot(tableDiagnosis, col="lightblue", main="Distribution of Diagnosis", xlab="Diagnosis", ylab="# of Observations")

age <- data$Age
tableAge <- table(age) 
# Age Distribution in data set
barplot(tableAge, col="khaki", main="Distribution of Ages", xlab="Age", ylab="# of Observations")

# Diagnosis vs Age
plot(diagnosis, data$Age, col="coral", xlab="Diagnosis", ylab="Age", main="Diagnosis vs Age")

# Distribution of Symptoms with Alzheimer's Diagnosis
(tableSymptoms <- c(
  sum(data$Smoking == "Yes", na.rm = TRUE),
  sum(data$FamilyHistoryAlzheimers == "Yes", na.rm = TRUE),
  sum(data$CardiovascularDisease == "Yes", na.rm = TRUE),
  sum(data$Diabetes == "Yes", na.rm = TRUE),
  sum(data$Depression == "Yes", na.rm = TRUE),
  sum(data$HeadInjury == "Yes", na.rm = TRUE),
  sum(data$Hypertension == "Yes", na.rm = TRUE),
  sum(data$MemoryComplaints == "Yes", na.rm = TRUE),
  sum(data$BehavioralProblems == "Yes", na.rm = TRUE),
  sum(data$Confusion == "Yes", na.rm = TRUE),
  sum(data$Disorientation == "Yes", na.rm = TRUE),
  sum(data$PersonalityChanges == "Yes", na.rm = TRUE),
  sum(data$DifficultyCompletingTasks == "Yes", na.rm = TRUE),
  sum(data$Forgetfulness == "Yes", na.rm = TRUE)
))

names <- c(
  "Smoking", "FamilyHistoryAlzheimers", "CardiovascularDisease", "Diabetes",
  "Depression", "HeadInjury", "Hypertension", "MemoryComplaints", "BehavioralProblems",
  "Confusion", "Disorientation", "PersonalityChanges", "DifficultyCompletingTasks",
  "Forgetfulness"
)
length(names)
length(tableSymptoms)
par(mar = c(11, 4, 4, 2)) #Increases bottom margin of graph window in R Studio
barplot(tableSymptoms, names.arg=names, las=2, col="lightgrey", main="Distribution of Symptoms with Diagnosis", ylab="Frequency")

#####

#2. Classification Pr#2. Classification Pr#2. Classification Procedures
  # Rpart Model
    # Show Confusion Matrix from Decision Tree

# Splits data into Training and Testing
(data_split <- initial_split(data, prop=0.80))

# Training data breakdown
training_data <- training(data_split)
summary(training_data)
prop.table(table(training_data$Diagnosis))

# Testing data breakdown
testing_data <- testing(data_split)
summary(testing_data)
prop.table(table(testing_data$Diagnosis))

# Constructs rpart tree using default cp at 0.0001
tree <- rpart(Diagnosis ~ ., data= training_data, method = 'class', cp=0.0001)
rpart.plot(tree, fallen.leaves = FALSE)
printcp(tree)

# Finding best cp to prune the tree
min_error_row <- which.min(tree$cptable[, "xerror"])
(best_cp <- tree$cptable[min_error_row, "CP"])
# Best cp = 0.003355705

# Creating a pruned tree
pruned_tree <- prune(tree, cp=best_cp)
rpart.plot(pruned_tree, fallen.leaves = FALSE)
printcp(pruned_tree)

# Confusion Matrix with Regular Tree (Predicting Training Data)
predict_tree <- predict(tree, testing_data, type="class")
(confusion_Matrix_tree <- table(predict_tree, testing_data$Diagnosis))
caret::confusionMatrix((confusion_Matrix_tree), positive="Yes")
# Accuracy = 0.9442

# Confusion Matrix with Pruned Tree (Predicting Training Data)
predict_prunedtree <- predict(pruned_tree, testing_data, type="class")
(confusion_Matrix_prunedtree <- table(predict_prunedtree, testing_data$Diagnosis))
caret::confusionMatrix((confusion_Matrix_prunedtree), positive="Yes")
# Accuracy = 0.9442

# K-Fold Validation with K=10
control <- trainControl(method='cv', number=10, savePredictions=TRUE)
(rpart_cv <- train(Diagnosis ~ ., data=training_data, method='rpart', trControl = control))
(best_cp_in_cv <- rpart_cv$results[which.max(rpart_cv$results[, 'Kappa']), "cp"])

# Creating final tree with best cp in Cross Validation
(finaltree <- rpart(Diagnosis ~., data = training_data, method = 'class', control = rpart.control(cp=best_cp_in_cv)))
printcp(finaltree)
rpart.plot(finaltree, fallen.leaves = FALSE)

# Predicting with the 10 Fold Cross Validation Tree (Predicting Testing Data)
predict_finaltree <- predict(finaltree, testing_data, type="class")
(confusion_finaltree <- table(predict_finaltree, testing_data$Diagnosis))
caret::confusionMatrix((confusion_finaltree), positive="Yes")
# Accuracy = 0.8302 

# Naive Bayes Model
# Use classification variable and predictors chosen from rpart
nb_model <- NaiveBayes(Diagnosis~FunctionalAssessment+MMSE+ADL, data = training_data)
nb_pred <- predict(nb_model, testing_data)
# Show Confusion Matrix
tab_nb <- table(nb_pred$class, testing_data$Diagnosis)
caret::confusionMatrix(tab_nb, positive="Yes")
# Compare accuracies
cat("rpart Accuracy: 0.8302")
cat("Naive Bayes Accuracy: 0.7814")
