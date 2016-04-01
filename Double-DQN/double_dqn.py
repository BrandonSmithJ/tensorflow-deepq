import sys
sys.path.append('tensorflow-deepq')

from tf_rl.controller.discrete_deepq import DiscreteDeepQ
import tensorflow as tf
import random
import numpy as np

class DDQN(DiscreteDeepQ):

	target_update_frequency = 1e4

	def create_variables(self):
		self.target_network_update_rate = 1.0
		with tf.variable_scope("evaluation_network"):
			self.eval_q_network = self.q_network.copy(scope="evaluation_network")

		self.target_network_update = []
		self.prediction_error = []
		self.train_op = []

		self.observation        	= tf.placeholder(tf.float32, (None, self.observation_size), name="observation")
		self.next_observation       = tf.placeholder(tf.float32, (None, self.observation_size), name="next_observation")
		self.next_observation_mask  = tf.placeholder(tf.float32, (None,), name="next_observation_mask")
		self.rewards                = tf.placeholder(tf.float32, (None,), name="rewards")
		self.action_mask            = tf.placeholder(tf.float32, (None, self.num_actions), name="action_mask")

		# Update both policy & evaluation networks
		for sc, q_network in [('evaluation',self.eval_q_network), ('policy', self.q_network)]:
			with tf.variable_scope(sc) as scope:
				target_q_network = self.q_network.copy(scope=sc)#+"_target_network")

				# FOR REGULAR ACTION SCORE COMPUTATION
				with tf.name_scope("taking_action"):
					action_scores      = tf.identity(q_network(self.observation), name="action_scores")
					tf.histogram_summary(sc + "action_scores", action_scores)
					self.predicted_actions  = tf.argmax(action_scores, dimension=1, name="predicted_actions")

				with tf.name_scope("estimating_future_rewards"):
					# FOR PREDICTING TARGET FUTURE REWARDS
					next_action_scores 	= tf.stop_gradient(target_q_network(self.next_observation))
					tf.histogram_summary(sc + "target_action_scores", next_action_scores)
					double_q_scores 	= tf.stop_gradient(self.eval_q_network(self.next_observation))
					double_q_actions 	= tf.reshape(tf.reduce_max(double_q_scores, reduction_indices = [1,]), [-1,1])
					double_q_onehot 	= tf.to_float(tf.equal(double_q_scores, double_q_actions))

					target_values  = tf.reduce_max(tf.mul(next_action_scores, double_q_onehot), reduction_indices=[1,])#, reduction_indices=[1,])
					future_rewards      = self.rewards + self.discount_rate * (target_values * self.next_observation_mask)

				with tf.name_scope("q_value_prediction"):
					# FOR PREDICTION ERROR
					masked_action_scores 		= tf.reduce_sum(action_scores * self.action_mask, reduction_indices=[1,])
					temporal_diff             	= masked_action_scores - future_rewards
					self.prediction_error.append( tf.reduce_mean(tf.square(temporal_diff)) )
					tf.scalar_summary(sc + "prediction_error", self.prediction_error[-1])
					gradients                   = self.optimizer.compute_gradients(self.prediction_error[-1])
					
					# Clip gradients
					for i, (grad, var) in enumerate(gradients):
						if grad is not None:
							gradients[i] = (tf.clip_by_norm(grad, 5), var)
					# Add histograms for gradients.
					for grad, var in gradients:
						tf.histogram_summary(var.name, var)
						if grad:
							tf.histogram_summary(var.name + '/gradients', grad)
					self.train_op.append(gradients)
		
		# Apply computed gradients to each (simultaneous updating)
		self.train_op = tf.group(*([self.optimizer.apply_gradients(gradients) for gradients in self.train_op]))
		self.prediction_error = tf.group(*self.prediction_error)
		for scope, q_network in [('evaluation',self.eval_q_network), ('policy', self.q_network)]:
			with tf.variable_scope(scope):

				# UPDATE TARGET NETWORK
				with tf.name_scope("target_network_update"):
					for v_source, v_target in zip(q_network.variables(), target_q_network.variables()):
						# this is equivalent to target = (1-alpha) * target + alpha * source
						update_op = v_target.assign_sub(self.target_network_update_rate * (v_target - v_source))
						self.target_network_update.append(update_op)

		self.target_network_update = tf.group(*self.target_network_update)
		#self.summarize = tf.merge_all_summaries()
		self.no_op1    = tf.no_op()



	def training_step(self):
		"""Pick a self.minibatch_size exeperiences from reply buffer
		and backpropage the value function.
		"""
		if self.number_of_times_train_called % self.train_every_nth == 0:
			if len(self.experience) <  self.minibatch_size:
				return

			# sample experience.
			samples   = random.sample(range(len(self.experience)), self.minibatch_size)
			samples   = [self.experience[i] for i in samples]

			# bach states
			states         = np.empty((len(samples), self.observation_size))
			newstates      = np.empty((len(samples), self.observation_size))
			action_mask    = np.zeros((len(samples), self.num_actions))

			newstates_mask = np.empty((len(samples),))
			rewards        = np.empty((len(samples),))

			for i, (state, action, reward, newstate) in enumerate(samples):
				states[i] = state
				action_mask[i] = 0
				action_mask[i][action] = 1
				rewards[i] = reward
				if newstate is not None:
					newstates[i] = newstate
					newstates_mask[i] = 1
				else:
					newstates[i] = 0
					newstates_mask[i] = 0

			calculate_summaries = self.iteration % 100 == 0 and \
					self.summary_writer is not None

			cost, _, summary_str = self.s.run([
				self.prediction_error,
				self.train_op,
				self.summarize if False and calculate_summaries else self.no_op1,
			], {
				self.observation:            states,
				self.next_observation:       newstates,
				self.next_observation_mask:  newstates_mask,
				self.action_mask:            action_mask,
				self.rewards:                rewards,
			})

			if self.iteration % self.target_update_frequency == 0:
				self.s.run(self.target_network_update)

			if calculate_summaries and False:
				self.summary_writer.add_summary(summary_str, self.iteration)

			self.iteration += 1

		self.number_of_times_train_called += 1