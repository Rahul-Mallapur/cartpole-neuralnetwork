import gym
import random 
import numpy as np
import statistics
from collections import Counter

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

#setup the Cartpole environment
env = gym.make("CartPole-v0")
env.reset()


#----------Explore CartPole-------------#
#exploring the observations, rewards, actions
def explore_cartpole():
	for i_episode in range(2):
	    observation = env.reset()
	    for t in range(100):
	        env.render()
	        print(observation)
	        action = env.action_space.sample()
	        observation, reward, done, info = env.step(action)
	        print("Action: ", action, "Rewards", reward)
	        if done:
	            print("Episode finished after {} timesteps".format(t+1))
	            break

#explore_cartpole()            

#----------Collect Training Data-------------#
#collect data from successful games by running x games
#successful would be say, lasting more than 100 frames
num_games = 20000
num_episodes = 201 #game would end at 200 episodes
min_score = 75

def initial_games():

	train_data = []
	train_scores = []

	#running our initial set of games
	for _ in range(num_games):
		game_data = []
		prev_obs = []
		score = 0

		#running the game, frame by frame
		for _ in range(num_episodes):
			#choosing actions: randomly
			action = random.randrange(0,2)
			observation, reward, done, info = env.step(action)

			if len(prev_obs) > 0: 
				game_data.append([prev_obs, action])

			prev_obs = observation
			score += reward

			if done:
				#print("Score was: ", score)
				break

		#if the score was above the threshold
		#we will save the game in our training data
		#hence training on the better games
		if score >= min_score :
			train_scores.append(score)
			#converting the data into one-hot output		
			for i in game_data:			
				if i[1] == 0:
					output = [1, 0]
				else:
					output = [0, 1]
				
				train_data.append([i[0], output])

		env.reset()

	return train_data


#----------Build the FC NN model-------------#
#building a simple multi-layer fully connected model
#this model can be generally used to play games like cartpole
#would try training the model on other games in OpenAI environment

def nn_model(input_size):

    network = input_data(shape=[None, input_size, 1], name='input')

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=1e-3, loss='categorical_crossentropy', name='targets')
    model = tflearn.DNN(network, tensorboard_dir='log')

    return model



#----------Train the model-------------#
def train_model(train_data, model=False):

	x = np.array([i[0] for i in train_data]).reshape(-1, len(train_data[0][0]),1)
	y = [i[1] for i in train_data]

	if not model:
		model = nn_model(input_size = len(x[0]))

	model.fit({'input': x}, {'targets': y}, n_epoch = 5, snapshot_step=500, 
		show_metric = True, run_id = 'openai_learning')
	return model

train_data = initial_games()
#print("Size of training data",len(train_data))

model = train_model(train_data)

#----------Predict actions for the games-------------#
num_final_games = 10
target_episodes = 201
all_rewards = []
all_actions = []

for _ in range(num_final_games):
	total_score = 0
	prev_obs = []
	env.reset()

	for _ in range(target_episodes):

		#env.render()

		#instead of randomly choosing the action, predict the actions
		if len(prev_obs) == 0:
			action = random.randrange(0,2)
		else:
			action = np.argmax(model.predict(prev_obs.reshape(-1,len(prev_obs),1))[0])
		
		all_actions.append(action)

		#let's run the game
		observation, reward, done, info = env.step(action)
		
		prev_obs = observation
		total_score += reward

		if done: 
			break

	all_rewards.append(total_score)

#----------Print results-------------#
print('Average reward:',np.mean(all_rewards), '+-', np.std(all_rewards))
print('Max reward:', max(all_rewards))
