------------------------ Thesis Abstarct ---------------------------------------

"With the advent of deep learning, machine learning community has started to
move from handcrafted features to learned ones. These learned features often
outperform manually crated features. Learning to optimize follows in the same
footsteps and tries to learn an optimizer. In this thesis we present two approaches
in the same direction, Learning to Optimize With Normalized Inputs"
and Multiscale Adam". In both approaches we train meta optimizers such that
given gradient information as input they output a gradient step. The first approach,
Learning to Optimize With Normalized Inputs" uses normalized history
of gradients as inputs. We explore various configurations of this approach e.g.
effect of using different history sizes. We also compare the learned optimizer
with Adam and RMSProp. Our trained optimizer outperforms both Adam and
RMSProp on the neural network it was trained on. Furthermore when applied
on an unseen neural network (Cifar10 ), the learned optimizer shows competitive
performance displaying its ability to generalize. The other approach, Multiscale
Adam" uses Adam running at different timescales as inputs and outputs a gradient
step that is a weighted average of its Adam inputs. We then compare the
performance of the learned optimizer with its individual Adam inputs. We do
thorough testing of our approach on a small neural network on Mnist, we see that
in most cases our trained optimizer outperforms its Adam inputs. Our trained
optimizer also generalizes to other unseen networks, e.g. Cifar10, with success.
In both approaches we start from simple models, test their capabilities, gradually
make them more complex and report the results. Finally, we also discuss the
limitations of both approaches as well."
-------------------------------------------------------------------------------------

slides.pdf contains the thesis defense presentation, while the "Learning to Optimize Deep Neural Networks.pdf" is the main thesis script. 
Forewarning the code is a hot mess and contains stuff that works along with a lot of stuff that I tried but didn't work very well.
