first attempt:
- copied code from example 4, adjusted input shape to match image size
- cifar10_rwd1.0e-01_rdp0.4_rbn1_daug0_cnn_net
- test-accuracy 55

26/03
- attempted to emulate vgg architecture `vgg11_net`
- full model wouldn't run due to memory
- reduce size of dense layers from 4096 to 1024
- 77/55 accuracy, 74/63 accuracy

26/03
- made cnn with column C from vgg paper `vgg16_net_small`
- 67/53 accuracy, 90/68 accuracy, 97/66
- overfitting after ~70 epochs

26/03
- added regularisation the same as in example 4 -> `vgg_net_small_with_regularization`
- 93/74 after 150 epochs
- balanced accuracy 68

27/03
- 
