import utils
import train
import numpy as np

def main():

	chromosome = 'arm_2L'
	inputfile = '../../H3K27me3_HiChIP_combined.hic'
	scalerate = 25
	outmodel = 'mymodel/v4model'
	startmodel = None
	startepoch = 0

	highres = utils.matrix_extract(chromosome, chromosome, 10000, inputfile)
	highres = np.array(highres.tolist())

	print('dividing, filtering and downsampling files...')

	highres_sub, index = utils.train_divide(highres)

	print(highres_sub.shape)

	lowres = utils.genDownsample(highres,1/float(scalerate))
	lowres_sub,index = utils.train_divide(lowres)
	print(lowres_sub.shape)

	print('start training...')
	train.train(lowres_sub,highres_sub,outmodel, startmodel, startepoch,scalerate)


	print('finished...')

if __name__ == '__main__':
    main()