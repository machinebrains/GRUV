def get_neural_net_configuration():
	nn_params = {}
	nn_params['sampling_frequency'] = 44100
	#Number of hidden dimensions.
	#For best results, this should be >= freq_space_dims, but most consumer GPUs can't handle large sizes
	nn_params['hidden_dimension_size'] = 2048
	#The weights filename for saving/loading trained models
	nn_params['model_basename'] = './weights/MusicLibraryNPWeights'
	#The model filename for the training data
	nn_params['model_file'] = './datasets/YourMusicLibraryNP'
	#The dataset directory
	nn_params['dataset_directory'] = '../Music/'
	return nn_params
