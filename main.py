import utils
import train
import numpy as np

def main():

	chromosome = 'arm_2L'
	val_chromosome = 'arm_2R'
	inputfile = '../../H3K27me3_HiChIP_combined.hic'
	scalerate = 25
	outmodel = 'mymodel/gan'
	startmodel = None
	startepoch = 0
	binsize = 10000

	highres = utils.matrix_extract(chromosome, chromosome, binsize, inputfile)
	print('dividing, filtering and downsampling files...')

	highres_sub, index = utils.train_divide(highres)
	print(highres_sub.shape)

	lowres = utils.genDownsample(highres,1/float(scalerate))
	lowres_sub,index = utils.train_divide(lowres)
	print(lowres_sub.shape)

	val_hires = utils.matrix_extract(val_chromosome, val_chromosome, binsize, inputfile)
	val_hires_sub, _ = utils.train_divide(val_hires)
	print('validate set shape:', val_hires_sub.shape)

	val_lowres = utils.genDownsample(val_hires, 1/float(scalerate))
	val_lowres_sub, _ = utils.train_divide(val_lowres)

	print('start training...')
	train.train(lowres_sub,highres_sub,val_lowres_sub, val_hires_sub,outmodel, startmodel, startepoch,scalerate)


	print('finished...')

if __name__ == '__main__':
    main()