import csv
import pandas as pd
import numpy as np
from collections import Counter
#from sklearn.linear_model import LinearRegression
#import matplotlib.pyplot as plt  # To visualize


if __name__ == "__main__":
	survey_file = "2021_Fall_Spatial_Reasoning.csv"
	question_mapping_file = "qualtrics_mapping.csv"
	csv_file = open(question_mapping_file)
	csv_data=csv.DictReader(csv_file)
	mapping_data = {}
	for line in csv_data:
		mapping_data[line["question"]]=line
	csv_file.close()
	survey_csv = open(survey_file)
	csv_file =csv.DictReader(survey_csv)
	user_data = []
	question_stats_scores = {}
	user_incorrect = Counter()

	for line in csv_file:

		c_user_data = {}
		c_user_list = []
		q_index = 0
		for question in mapping_data.keys():
			if question not in question_stats_scores:
				question_stats_scores[question]=[]
			response_correctness = str(line[question]) == str(mapping_data[question]["same"])
			last_click = float(line["Q"+str(mapping_data[question]["time_question"]) + "_Last Click"])
			page_submit = float(line["Q"+str(mapping_data[question]["time_question"]) + "_Page Submit"])
			if last_click == 0.0:
				last_click=10.0
			question_stats_scores[question].append(last_click)
			c_user_list.append([str(response_correctness), str(last_click),str(float(mapping_data[question]["angle"])),str(mapping_data[question]["same"])])
			#c_user_data[mapping_data.keys().index(question)] = [response_correctness,last_click,float(mapping_data[question]["angle"]),mapping_data[question]["same"]]
			c_user_data[q_index] = [response_correctness,page_submit]
			if not(response_correctness):
				user_incorrect[len(user_data)]+=1
			q_index += 1
		user_data.append(c_user_data)
		#c_user_data_df=pd.DataFrame.from_dict(c_user_data, orient="index")
		#linear_regressor = LinearRegression()  # create object for the class
		#X = c_user_data_df[[1, 2]]
		#Y = c_user_data_df[0]
		#regr = linear_regressor.fit(X, Y)  # perform linear regression
		#Y_pred = linear_regressor.predict(X)  # make predictions
		#print(Y_pred)
		##print('Intercept: \n', regr.intercept_)
		##print('Coefficients: \n', regr.coef_)
		#plt.scatter(X, Y)
		#plt.plot(X, Y_pred, color='red')
		#plt.show()
		#fig = px.scatter(c_user_data_df[1], c_user_data_df[0])
		#fig.show()
		#print(c_user_data_df)
	user_z_scores = {}
	q_index = 0
	for question in mapping_data.keys():
		c_mean = np.mean(question_stats_scores[question])
		c_std = np.std(question_stats_scores[question])
		for user in user_data:
			c_score = (user[q_index][1]-c_mean)/c_std
			c_user = user_data.index(user)
			if c_user not in user_z_scores:
				user_z_scores[c_user]=[]
			user_z_scores[c_user].append(c_score)
		q_index += 1

	out = open("data_output.csv", "w+")
	out.write('user,Q1,Q2,Q3,Q4,Q5,Q6,Q7,Q8,Q9,Q10,Q11,Q12,Q13,Q14,Q15,Q16,Q17,Q18,Q19,Q20,user_incorrect,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,T18,T19,T20,C1,C2,C3,C4,C5,C6,C7,C8,C9,C10,C11,C12,C13,C14,C15,C16,C17,C18,C19,C20,average_score\n')
	user_index = 0
	for user in user_data:
		out.write(",".join([str(user_index)]+[str(a) for a in user_z_scores[user_index]]+ [str(user_incorrect[user_index])] + [str(user_data[user_index][map_i][0]) for map_i in range(len(mapping_data.keys()))]+ [str(user_data[user_index][q_i][1]) for q_i in range(len(mapping_data.keys()))] + [str(sum(user_z_scores[user_index])/len(user_z_scores[user_index]))]) + "\n")
		user_index += 1 
	out.close()
