################################################################################
#This code provides an in-depth analysis of athlete performance data using #
#standard graphical and numerical summaries, classification models and model #
#selection. Graphical summaries include histograms plotting medal winning status #
#against age and training hours and a scatterplot of medal winning status against #
#training hours. Numerical summaries include tables displaying medal winning #
#status against each of the following categorical covariates: previous medal #
#winning status, country group, main sponsor and top10 (i.e. rank status). Model #
#selection techniques include use of AIC to select the best (additive) generalised #
#linear model and use of 10-fold cross validation to select best linear and #
#kernel support vector classifier based on margin width and tuning parameter #
#gamma using predictors age and training hours. Classification models include #
#logistic regression fits of medal winning status on all covariates (including #
#and excluding country and main sponsor), linear and quadratic discriminant #
#analysis models using the variables that minimise AIC when performing #
#logistic regression and support vector machines using quantitative covariates #
#age and training hours.#

#The computations are made using R2025.09.2+418 and all libraries downloaded #
#prior to certain computations correspond to packages built under R version #
#4.2.3. The majority of code used in fitting and selecting models are based on #
#coding examples in the core textbook for the Machine Learning and Deep Learning #
#module "An Introduction to Statistical Learning with Applications in R: Lab: #
#Logistic Regression, LDA, QDA and KNN" and "An Introduction to Statistical #
#Learning with Applications in R: Lab: Support Vector Machines", Springer 2013, #
#with permission from authors James.G, D.Witten, T.Hastie and R.Tibshirani.#
################################################################################

### Data Quality Checking & Exploratory Data Analysis ###

## Users should refer to the below code when performing data quality checks #
# and carrying out exploratory data analysis to confirm appropriateness #
# of the model selection techniques and statistical learning methods used in #
#this data analysis. ##

ath<-read.csv("athletes_medal.csv")
#Assigns shorter name to the data file used for below analysis. Note that R will #
#assume the athletes data file is saved down in the same directory as the one set #
#for the session.#
medal<-ath$medal
#Assign unique name to response variable in data set.#
sum(is.na(medal))
#Checks for missing values of response denoted NA.#
sum(is.na(ath$age))
#Checks for missing values of age covariate denoted NA.#
sum(is.na(ath$hours_training))
#Checks for missing values of training hours covariate denoted NA.#
max(ath$age)
#Checks for any observations that exceed the age range to check for error codes #
#99. #
table(medal)
#Creates table regarding number of athletes who did or didn't win a medal. Upon #
#running code, the total numbers do add up to 2,349 so all athletes are classed #
#as either Yes or No.#
table(ath$previous_medal)
#Creates table regarding numbers of athletes who did or didn't win a medal in #
#the previous competition. Again, numbers of observations across both classes #
#add up to 2,349.#
table(ath$top10)
#Creates table regarding numbers of athletes who did or didn't rank among top #
#10 athletes. Again, numbers of observations across both classes #
#add up to 2,349.#
table(ath$country)
#Creates table regarding country group that each athlete belongs to. 816 under #
#group A plus 761 under group B and 772 under group C add up to 2,349 as required.#
table(ath$main_sponsor)
#Creates table regarding main sponsors of athletes. A=763, N=693, P=446 and U=447 #
#lead to a total figure of 2,349 as required. Hence, no data quality issues of #
#categorical covariates.#
library(ggplot2)
#Loads ggplot2 library which stores a variety of functions used in graphical #
#analysis.#
ggplot(ath)+aes(x=ath$hours_training,y=ath$medal)+geom_point()+theme_minimal()+geom_smooth(method=lm,se=FALSE)
#Fits linear model of the binary response variable on the training hours using plotted points.#
#From this we can see that, because medal only takes values 0 and 1, a linear model may not be suitable.#
#If this line was steeper, predicted values of medal could more easily exceed the [0,1] range.#
#Since we are interested in measuring probabilities of winning a medal, a logistic regression #
#model would be more suitable for this analysis.#
#But there is a trend; athletes who work more hours have higher chances of winning a medal. Those #
#who work around 14 hours a week have a 25% chance of winning a medal, which increases to 50% if #
#they work an additional 6 hours a week.#
ggplot(ath)+aes(ath$hours_training)+geom_histogram(fill="royalblue",color="grey75",binwidth=0.5)+theme_bw()+facet_wrap(~medal)
#Plots 2 different histograms of the weekly number of hours worked for different values of the response.#
#For this plot, based on values of the covariate, I have used a binwidth of 0.5 for ease of identifying a trend.#
#The distribution of working hours for those who didn't win a medal is positively skewed, and negatively #
#skewed for those who did. Trends show that more athletes who didn't win a medal worked fewer hours as #
#opposed to longer hours, and vice versa for those who did win a medal.#
table(ath$previous_medal,ath$medal)
#Produces table comparing previous medal wins against current medal wins.
410/(903+410)
442/(442+594)
#Compute proportions of medal winners who did or didn't win a previous medal. Roughly 31% of medal winners #
#who didn't win a previous medal won a medal in the current competition, increasing to 43% for previous #
#medal winners. Both proportions being under 50% could result in a negative coefficient for previous_medal #
#when logistic regression is performed.
table(ath$top10,ath$medal)
#Produces table comparing those who are ranked among the top 10 athletes against those who won a #
#medal in the current competition.#
534/(1103+534)
318/(318+394)
#Computes proportions of medal winners who are or aren't ranked among the top 10 athletes. Roughly 33% #
#of medal winners aren't ranked top 10, increasing to roughly 45% for those who are. Again, both are #
#under 50% so could result in a negative coefficient estimate for top10.#
ggplot(ath)+aes(age)+geom_histogram(fill="royalblue",color="grey75",binwidth=1)+theme_bw()+facet_wrap(~medal)
#Plots 2 different histograms of age for different values of the response. Trends show that as age #
#increases, the number of athletes who don't win a medal increase, and vice versa for those who do.#
#Hence, when the model is fitted, we should see that if age increases by one year, the probability #
#of winning a medal reduces.#
table(ath$country,ath$medal)
#Produces table comparing medal wins against country group
277/(277+539)
285/(285+476)
290/(482+290)
#Computes proportions of athletes in each country group who won a medal. All proportions are less than #
#50%, but a higher proportion of group B and C athletes won a medal than group A athletes, so should #
#groups B and C appear as indicator variables in the model output, we could see positive coefficient estimates.#
table(ath$main_sponsor,ath$medal)
#Produces table comparing medal wins against main sponsors.
285/(285+478)
243/(243+450)
156/(156+290)
168/(168+279)
#Computes proportions of athletes with each sponsor who won a medal. Sponsor U has the highest proportion #
#of medal winners, hence could result in a positive coefficient if sponsor=U is shown as an indicator #
#variable in the model output. But for country group and sponsor, proportions of winners are all under #
#50%, hence both covariates could reduce the chances of winning in the fitted model.#

### Model Selection + Fitting Using Logistic Regression & Prediction Accuracy ###

##Users should refer to the below code when performing model selection and assessing #
#prediction accuracy using a logistic regression fit.##

library(MASS)
#Loads library MASS without which model selection using AIC, LDA and QDA cannot be performed.#
ath_fullmod<-glm(ath$medal~ath$previous_medal+ath$top10+ath$country+ath$main_sponsor
                 +ath$age+ath$hours_training,data=ath,family=binomial)
#Assigns name to logistic regression fit of medal on all 6 covariates. This represents #
#the full additive model.#
summary(ath_fullmod)
#Computes summary of full logistic regression model of medal on all covariates. By observing #
#signs of coefficients, we see that some of the prior hypotheses are violated, with previous #
#medal wins and top 10 ranks increasing chances of winning a medal by 49% and 51%, respectively.#
#The positive coefficients could be due to the fact that, with reference to the relevant tables, #
#there were higher proportions of medal winners who were previous medal winners compared to those #
#who weren't, as well as higher proportions of medal winners who rank among the top 10 compared #
#to those who don't.#
#Hypotheses regarding the quantitative covariates hold, with a 1 year increase in age reducing #
#chances of winning by 5.4% and one additional hour per week of training increasing chances of #
#winning by 18%. Also as predicted, athletes belonging to country groups B or C or having main #
#sponsor U increase their chances of winning, but p-values indicate little significance of #
#country and sponsor on the chances of winning, hence this conclusion may not hold.#
ath_bestmod<-stepAIC(ath_fullmod,direction="both")
#Performs step-wise regression starting with the full additive model of medal on the covariates #
#using AIC criterion. From the output, we can conclude that previous medal wins, top 10 ranks, #
#age and weekly training hours all have a significant influence on the chances of winning a #
#medal in the current competition.#
summary(ath_bestmod)
#Computes summary of model with lowest AIC on the variables with the most significant influence #
#on winning a medal. Similar conclusions to full model output can be drawn based on coefficient #
#signs.#
ath.probs=predict(ath_bestmod,type="response")
#Assigns name to probabilities of winning a medal predicted using the model with the lowest AIC.#
ath.probs[1:20]
#Computes first 20 predicted probabilities of winning a medal.#
athbest_pred=rep("No",2349)
athbest_pred[ath.probs>0.5]="Yes"
#Converts predicted probabilities into predicted class labels for the medal response variable.#
table(athbest_pred,ath$medal)
#Produces confusion matrix to determine accuracy of classification of observations. Diagonal #
#elements indicate that of the 2,349 athletes, 1,534 were correctly classified as either not #
#winning a medal or winning a medal based on predicted and actual classes.#
(1367+167)/2349
#Computes rate at which athletes were accurately classified as not winning or winning a medal.#
#The output indicates a prediction accuracy rate of 65.3%, indicating that the best logistic #
#regression model of medal on the significant covariates is better than random guessing. It #
#also indicates a training error rate of 100% - 65.3% = 34.7%. Below we will use cross-validation #
#to better assess the accuracy of this model by working out a test error rate, as this model has #
#been fit on all 2,349 athletes as opposed to a set of training observations, hence a possible #
#underestimate of the true error rate.#
ath.train<-(ath$hours_training<19)
#Creates vector of training observations represented by athletes who trained for under 19 hours #
#a week.#
ath_19<-ath[!ath.train,]
#Creates vector of held out observations represented by athletes who trained for more than 19 hours #
#per week. These observations will be used to test the best additive model of medal on the covariates.#
dim(ath_19)
#Computes the number of dimensions of the matrix represented by the test observations. The first figure #
#represents the number of observations in the validation set, which in this case is 410. The second #
#figure represents the number of variables in the data set, which is 7 for the athletes data set - 1 #
#response and 6 covariates.#
medal_19=ath$medal[!ath.train]
#Creates vector of test responses.#
ath_trainmod<-glm(medal~previous_medal+top10+age+hours_training,
                  data=ath,family=binomial,subset=ath.train)
#Fits logistic regression model of medal on the 4 significant covariates using only training #
#observations.#
ath_trainprobs=predict(ath_trainmod,ath_19,type="response")
#Creates vector of predicted probabilities of winning a medal using model fit on training set.#
athtrain_pred=rep("No",410)
athtrain_pred[ath_trainprobs>0.5]="Yes"
#Converts predicted probabilities into class labels for athletes represented by test observations.#
table(athtrain_pred,medal_19)
#Produces confusion matrix to determine accuracy of classification of test observations. From this, #
#we can deduce that of the 410 athletes who trained longer than 19 hours a week, 224 were correctly #
#classified as either winning or not winning a medal based on predicted and actual classes.
(147+77)/(147+77+66+120)
#Computes rate at which "test" athletes were correctly classified. From the output, we can see that #
#using only test observations, the model was able to correctly predict an athlete's medal winning #
#status 54.6% of the time, giving rise to a test error rate of 45.4%. This error rate is higher than #
#when we fit a logistic regression model on all 2,349 athletes, but the prediction accuracy rate of #
#55.1% is still better than random guessing. The higher error rate is expected, given that the model #
#is being tested against different values/sets of observations as opposed to the whole data set. #
#However, by assessing prediction accuracy for both models, we can conclude that when it comes to #
#performing logistic regression of medal on the 4 significant predictors highlighted, the model that #
#uses all observations yields a higher accuracy rate (albeit at the cost of it not being the true #
#error rate).#

## Model Fitting Using LDA and QDA ##

##Users should refer to the following code when assessing suitable discriminant #
#analysis fits to the athletes data. Upon computing the below code, users will #
#be able to determine the best model out of logistic regression, LDA and QDA by #
#comparing test error rates under each.#

ath.lda<-lda(medal~previous_medal+age+hours_training+top10,
             data=ath,subset=ath.train)
#Fits linear discriminant analysis model using covariates from the best model #
#selected using AIC and the observations representing athletes who worked less #
#than 19 hours a week.#
ath.lda
#Provides summary of LDA model output. The output suggests that 66.2% of "training" #
#athletes didn't win a medal, while the latter 33.8% did. The group means imply #
#similar trends to prior analyses; athletes who win a medal tend to be younger on #
#average than those who don't, train for more hours a week and tend to be previous #
#medal winners and/or ranked among the top 10 athletes. Coefficients of linear #
#discriminants give rise to the following equation: 1.12*previous_medalyes - #
#0.12*age + 0.34*hours_training + 1.28*top10yes. If the value of the discriminant #
#function is large, the LDA model will predict that an athlete will win a medal.#
ath.lda.pred=predict(ath.lda,ath_19)
#Uses LDA model fit using training observations to predict medal winning status for #
#test athletes.
athpred.class=ath.lda.pred$class
#Creates vector to obtain predicted medal winning statuses, referred to as class in #
#the predict function.#
table(athpred.class,medal_19)
#Produces confusion matrix comparing predicted and actual medal winning status #
#labels using only test observations. The results from the output are very similar #
#to when predictions were made using logistic regression, only an additional test #
#athlete has been misclassified. Therefore, by considering test errors, logistic #
#regression marginally outperforms linear discriminant analysis in terms of #
#prediction accuracy. By considering the posterior probability that an athlete #
#will not win a medal:#
sum(ath.lda.pred$posterior[,1]>0.75)
#Computes total number of athletes with a posterior probability of not winning a #
#medal greater than 75%, which is 0. A smaller threshold is thus required.
sum(ath.lda.pred$posterior[,1]>0.7)
#Computing this suggests that of the 410 athletes corresponding to the test #
#observations, 21 had a posterior probability of not winning a medal greater #
#than 70%. By considering the 70% and 75% thresholds, we can conclude that the #
#maximum possible posterior probability of not winning a medal based on the LDA #
#model is between 70% and 75%.
ath.qda<-qda(medal~previous_medal+age+hours_training+top10,
             data=ath,subset=ath.train)
#Fits quadratic discriminant analysis model using covariates from the best model #
#selected using AIC and the observations representing athletes who worked less #
#than 19 hours a week.#
ath.qda
#By computing this, we observe similar results to the LDA model output. But since #
#the discriminant function is not linear under QDA, linear discriminant #
#coefficients are no longer part of the output.
ath.qda.pred=predict(ath.qda,ath_19)
#Uses QDA model fit using training observations to predict medal winning status for #
#test athletes.
athpred.qda.class=ath.qda.pred$class
#See line 243.#
table(athpred.qda.class,medal_19)
#The output of this confusion matrix indicates that of the 410 test athletes, 216 #
#were correctly classified as either winning or not winning a medal under the QDA #
#model. Under LDA, 223 were correctly classified, whereas under logistic regression, #
#224 were correctly classified. Comparing these 3 methods, the logistic regression #
#fit yields the lowest test error rate, hence is a better fit to the athletes data #
#than discriminant analysis. By comparing discriminant analysis fits, LDA appears to #
#be a more suitable model than QDA, and may imply that for the entire athletes data #
#set, a linear decision boundary separating the medal winning classes may yield #
#higher accuracy. It may also imply that the variances of predictor values of #
#observations are similar in both classes.#
install.packages("dplyr")
library(dplyr)
#Installs and loads the dplyr package which stores a variety of functions that are #
#useful for data summary and manipulation (e.g. select, filter, group by, etc.).#
#Package was built under R version 4.2.3.
library(magrittr)
#Installs library that allows the pipe-operator %>% to be applied when executing code #
#that uses functions from the dplyr library. Users must note that it must be applied #
#after each line of code except the final line of code used in performing a data #
#summary or manipulation.#
ath %>%
  group_by(ath$medal) %>%
  summarise(var_age=var(ath$age),var_hours=var(ath$hours_training))
#In the 3 lines of code above, the athletes data set is being "piped" into the #
#expression that groups observations by medal winning status. The grouping expression #
#is then piped into an expression that calculates the conditional variances of the data #
#set's numeric predictors, separately for each medal class. The output confirms that #
#the variances of athletes' ages and weekly training hours are the same for each class, #
#thus reinforcing the conclusion that LDA provides a better fit for the athletes data #
#than QDA.#

## Model Fitting Using Support Vector Machines ##

##Users should refer to the following code when fitting support vector classifiers using #
#this data set. Note that the response variable in this data set is binary/2-class, which may #
#make support vector machines a suitable statistical learning method for this analysis.#
#For the purposes of this analysis, we will fit a support vector classifier of medal on #
#the 2 quantitative covariates.#

quant<-cbind(ath$age,ath$hours_training)
#Combines quantitative covariates age and training hours into a 2-column matrix.#
x=matrix(quant,ncol=2)
#Denotes 2-column matrix as x
y=ath$medal
#Denotes response variable in athletes data set as y.#
x[y=="Yes",]=x[y=="Yes"]*1
plot(x,col=(3-y))
#Plots covariates against response whilst assigning 2 different colours to the points #
#represented by the different classes. From this, we can see that age and training hours #
#are not linearly separable. In fact, because age is a discrete random variable and hours #
#is continuous, the plot displays 10 vertical lines of points. Further, the spread of points #
#representing each class may mean that the covariates won't be separable by any hyperplane, let #
#alone a straight line. This can be confirmed by the below lines of code.#
ath_dat=data.frame(x=x,y=as.factor(y))
#Creates data frame of values of the response and quantitative covariates. Since the response is #
#categorical, it is encoded as a factor variable.#
library(e1071)
#Loads library which stores functions that are useful for fitting support vector classifiers.#
#Corresponding package was built under version 4.2.3.#
ath_svm=svm(y~.,data=ath_dat,kernel="linear",cost=10,scale=FALSE)
#Fits support vector classifier using a linear hyperplane, a maximal margin represented by cost=10 #
#and no scaling of predictors to have mean 0 and standard deviation 1.#
plot(ath_svm,ath_dat)
#Plots quantitative covariates and support vector classifier on the same diagram. Red crosses #
#represent support vectors which are within the hard margin of the hyperplane representing the #
#support vector classifier. Non-medal winners are represented as points in the peach space while #
#medal winners are represented as points in the burgundy space. From the plot, we can only see #
#the points represented by non-medal winners, and hence no hyperplane separating the classes. But #
#this is being fit using all observations. Hence, we will try to fit a classifier using only the #
#test observations as our sample.#
quant_19<-cbind(ath_19$age,ath_19$hours_training)
x_19=matrix(quant_19,ncol=2)
y_19=ath_19$medal
x_19[y_19=="Yes"]=x_19[y_19=="Yes"]*1
plot(x_19,col=(3-y_19))
ath_19_dat=data.frame(x=x_19,y=as.factor(y_19))
ath_19_svm=svm(y~.,data=ath_19_dat,kernel="linear",cost=10,scale=FALSE)
plot(ath_19_svm,ath_19_dat)
#The above lines of code repeats the same process of fitting a support vector classifier but this #
#time using the test observations specified when performing logistic regression. The results look #
#better; we have a linear hyperplane separating medal winners from non-medal winners, and can see #
#that of the athletes who trained longer than 19 hours, a significant majority of the under-24s won #
#a medal. #
summary(ath_19_svm)
#The following summary of the classifier tells us that of the 410 test observations, 394 are support #
#vectors. This means 394 observations are contained within the soft margins of the hyperplane, but #
#from the previous plot, we can see that not only are a lot of these vectors on the wrong side of #
#the margin but also the wrong side of the hyperplane. By reducing cost to, say, 0.5, the margin #
#widens, but this could mean all test observations become support vectors. Hence, we will now #
#perform cross validation using higher costs (i.e. smaller margins) to see if any of them yield #
#better accuracy than cost=10.#
set.seed(1)
#Generate same set of random numbers.
ath.tune=tune(svm,y~.,data=ath_19_dat,kernel="linear",ranges=list(cost=c(5,10,20,30,40,50,60,70,80,90,100)))
#Performs cross validation using data on the 410 "test" athletes and different margin widths.#
summary(ath.tune)
#The output implies that 10-cross validation was performed for each margin width, and that cost=5 #
#results in the lowest cross-validation error rate of roughly 51%. In order to see if the error can #
#be reduced further, 10-fold cross-validation will be performed again but using wider margins.#
ath.tune.lower=tune(svm,y~.,data=ath_19_dat,kernel="linear",ranges=list(cost=c(0.001,0.01,0.1,0.5,1,5,10)))
summary(ath.tune.lower)
#This output implies cost=0.001 yields the lowest cross-validation error rate. The error rate has reduced #
#but only to around 48%. By predicting medal winning status labels for these test athletes using this cost, #
#the misclassification rate regarding the best model can be obtained.#
athsvm_bestmod=ath.tune.lower$best.model
#Assigns name to model with cost paramter 0.001.
summary(athsvm_bestmod)
#From this, we can see that the number of support vectors is the same for cost parameters 0.001 and 10.#
table(predict=predict(athsvm_bestmod,ath_19_dat),truth=ath_19_dat$y)
#In fitting the support vector classifier to all observations, no hyperplane could be seen and only #
#athletes who didn't win a medal were displayed. The output of the table indicates a similar situation; #
#all 410 test athletes were predicted to not win a medal, implying that when the classifier was fit, #
#all observations were on the side of the hyperplane representing non-medal wins. Of the 410 test #
#athletes, 213 were correctly classified.#
(410-213)/(213+197)
#Confirms error rate corresponding to best model selected using 10-fold cross-validation, #
#implying a 52% accuracy rate associated with the best linear hyperplane. Below is a #
#repeat of this process but using a radial kernel to confirm whether a linear or non-linear #
#boundary is suitable for separating the 2 classes for this subset of observations.#
svm_rad=svm(y~.,data=ath_19_dat,kernel="radial",gamma=1,cost=10)
plot(svm_rad,ath_19_dat)
#From the resulting plot, we can see that of those who trained more than 19 hours a week, #
#the non-linear hyperplane classifies younger medal winners as working closer to 20 hours #
#a week and older medal winners closer to the 19-hour threshold, with some older winners #
#working around 19.5 hours and some younger winners working close to 19 hours. This may #
#indicate that older and more experienced (and fitter) athletes require fewer training #
#hours to perform at their best in the competition and win a medal. The narrow cost #
#parameter of cost=10 may explain why the non-support vectors are visible, unlike #
#when the linear boundary was used. This may improve the accuracy of classification, # 
#as a narrower margin leads to lower bias of classification results.#
summary(svm_rad)
#From the output, we can see that there are fewer support vectors than when the linear #
#hyperplane was used, confirming that more observations are further away from the maximal #
#margins than when the SVM with a linear kernel was used. Again, cross-validation will #
#be used to determine the value of the tuning parameter and width of the soft margin #
#corresponding to the best support vector classifier.#
set.seed(1)
ath.tune.rad=tune(svm,y~.,data=ath_19_dat,kernel="radial",ranges=
                    list(cost=c(0.0001,0.01,1,100,10000),
                         gamma=c(0.5,1,2,3,4)))
summary(ath.tune.rad)
table(predict=predict(ath.tune.rad$best.model,ath_19_dat),truth=ath_19_dat$y)
#From the summary, we can see that the best classifier with a radial kernel is the #
#model where the cost (i.e. soft margin width) = 10000 and the tuning parameter #
#is 0.5. Optimal cost and gamma is equivalent to the narrowest margin and lowest #
#value of gamma used in performing 10-fold cross validation. With reference to #
#the formula for radial kernel, lower tuning parameters alone with increase the kernel, #
#thus widening the decision boundary. If the decision boundary is wide, it is more #
#likely to gain close proximity to observations in both classes, hence more nearby #
#observations will play a significant role in determining medal winning status. The #
#error rate for the best classifier is 38.5%, which is lower than when the linear #
#support vector machine was fit. From this, we can conclude that when it comes to #
#fitting a support vector classifier onto the 410 test observations for this data #
#set using the quantitative covariates, a non-linear kernel will improve prediction #
#accuracy for this subset of observations. More generally, it could be argued that #
#when it comes to using a supervised statistical learning method involving decision #
#boundaries designed to separate classes, non-linear/polynomial boundaries will be a #
#better fit for these 410 test athletes, but for the whole data set, a linear boundary #
#is more suitable (as observed when fitting discriminant analysis models to the data.)#

#From the confusion matrix, the comparison of predicted vs actual medal winning statuses #
#is more promising that when the linear SVM was fit; there are now non-zero predictions #
#regarding the number of test athletes who will win a medal. Confirming that the #
#true error rate is equivalent to the error rate for the best radial support #
#vector classifier:#
(74+61)/(74+61+139+136)
#It is not. In fact, it is less than the error rate obtained in cross-validation.#
#Nevertheless though, a similar conclusion regarding accuracy of prediction using #
#kernel SVMs for this subset of observations can be reached.#

### Users must note that, whilst the conclusion regarding the best type of SVM holds #
#for the specified test observations, it may not hold for the whole athletes data set.#
#Non-appearance of the support vector classifier when fitting linear SVM on entire #
#data set may imply that support vector machines are unsuitable for large data sets #
#such as the athletes. Studies of athletes' performance and demographic at a more #
#local level may improve suitability of SVMs to such data. Other reasons why SVMs #
#are unsuitable include use of a discrete numeric covariate (in this case, age) to #
#fit an SVM, as the plot of age against training hours meant it was difficult to #
#separate the non-medal-win and medal-win classes, thus making it harder to correctly #
#classify the observations. Another reason could include the 30%+ error rates, but #
#whether that reinforces unsuitability of SVMs to this data set is subject to what #
#is accepted as a reasonable error rate.#


