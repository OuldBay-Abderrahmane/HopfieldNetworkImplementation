# Analysis of capacity and robustess of Hebbian and Storkey Hopfield Networks

## Abstract

#### For each network size, run the previously defined experiment (10 runs with different initial  patterns) for different values of the parameter t. Include in your experiments values of t for which the system successfully converges to the initial pattern and others for which it does not. Estimate in this way the network capacity as a function of the network size and compare it with the theoretical asymptotic estimate.

#### In addition to the result tables, save capacity curves for your experiments (plotting fraction of retrieved patterns vs. t for each n). Make sure the figure is well formatted.

#### Save one plot with your empirical capacity curves including number of neurons vs. capacity (defined as #patterns that can be retrieved with >= 90% probability) for both learning rules and compare it to the theoretical estimate!

#### t = 2. Compute the network weights with the Hebbian rule. Progressively increase the initial perturbation by steps of 5%. At which point does the system stop converging to the initial pattern (in at least 9 out of 10 of your runs)? Repeat the experiment with the Storkey weights. Are the results the same as before? In addition to the result tables, save robustness curves for your experiments (plotting fraction of retrieved patterns vs. number of perturbations).

## Experiment results

### Capacity measure 

#### For each network size n, and each pattern number t (with 10 pattern numbers arranged around the theoretical capacity), we ran an experiment 10 times, giving us a retrieval rate out of 10 for states pertubed at 20%.

**Fraction of retrieved patterns for every t for every size of network for the hebbian learning rule**

![capacity-hebbian](https://user-images.githubusercontent.com/71345328/149383980-5de3e2d4-9bfc-4f90-bcb2-38bcf6964680.png)

***Fraction of retrieved patterns for every t for every size of network for storkey learning rule**

![capacity-storkey](https://user-images.githubusercontent.com/71345328/149383982-16e731bc-67ff-4248-b4f0-c70fb961449b.png)

**Hebbian learning rule experimental capacity**
|size       |   10 |   18 |   34 |   63 |   116 |   215 |   397 |   733 |   1354 |   2500 |
|----------:|-----:|-----:|-----:|-----:|------:|------:|------:|------:|-------:|-------:|
|empirical  |    2 |    2 |    3 |    4 |     8 |     9 |    15 |    25 |     32 |     55 |
|*theoretical|    1 |    2 |    3 |    5 |     *8 |    13 |    22 |    38 |     65 |    110 |

**Storkey learning rule experimental capacity**
|size       |   10 |   18 |   34 |   63 |   116 |   215 |   397 |   733 |   1354 |   2500 |
|----------:|-----:|-----:|-----:|-----:|------:|------:|------:|------:|-------:|-------:|
|empirical  |    3 |    4 |    5 |    0 |     0 |     0 |     0 |     0 |      0 |      0 |
|theoretical|    3 |    6 |   10 |   18 |    31 |    54 |    95 |   168 |    296 |    526 |


#### Analysis

For the Hebbian learning rule, it appears that as the size of the networks goes up, the empirical capacity seems to get further from their theoretical value, while beeing relatively close for smaller sizes. However, using the Storkey learning rule it seems that our network completely stops beeing able to retrieve patterns regardless of their number after the first three sizes. We don't know what caused this kind of error.

#### Conclusion

The Storkey and Hebbian learning rule cannot be compared capacity wise with our current dataset because of the lack of information on the capacity of a network following the Storkey rule. It can only be noted that the empirical capacities of the networks using the Hebbian rule only seem to be accurate with smaller sized networks. Further experiments with different percentage of perturbations and number of patterns could give us more informations on the Hebbian rule and especially the Storkey rule, knowing that the retrieval of patterns works for lower numbers of pattern for the bigger sizes (since the previous demonstations of our networks using the Storkey rule and 50 original patterns worked and the lowest number tested was around 200 patterns).


### Robustness measure

#### For each network of 2 patterns of size n, we ran several experiments increasing progressively the perburbation on a pattern in order to see how much perturbations on the original pattern could be done  and still allow retrieval.

**Fraction of retrived patterns for the percentage ofor every size of network for the hebbian learning rule**

![robustness-hebbian](https://user-images.githubusercontent.com/71345328/149384009-f21e07d8-86ef-44e5-b7f5-f7f841b94a0d.png)

**Fraction of retrived patterns for the percentage of perturbation for every size of network for the storkey learning rule**

![robustness-storkey](https://user-images.githubusercontent.com/71345328/149384012-fe68e34a-9985-4c21-94cc-02a0645a41b9.png)

**Hebbian learning rule experimental robustness**
|                |   10 |   18 |   34 |   63 |   116 |   215 |   397 |   733 |   1354 |   2500 |
|---------------:|-----:|-----:|-----:|-----:|------:|------:|------:|------:|-------:|-------:|
|pertubation rate| 0.25 |  0.3 | 0.35 |  0.5 |   0.5 |   0.6 |  0.55 |  0.65 |   0.65 |   0.65 |

**Storkey learning rule experimental robustness**
|                |   10 |   18 |   34 |   63 |   116 |   215 |   397 |   733 |   1354 |   2500 |
|---------------:|-----:|-----:|-----:|-----:|------:|------:|------:|------:|-------:|-------:|
|pertubation rate| 0.35 | 0.35 | 0.45 |  0.5 |  0.55 |  0.55 |   0.6 |   0.6 |    0.6 |   0.65 |

#### Analysis 
 
Regarding the robustness, it seems similar for both the Storkey and Hebbian learning rule when looking at the graphs, however, the table shows an overall very slight advantage for the Storkey learning rule (which can retrieve patterns with up to 10% more perturbation towards the smaller sized networks).

#### Conclusion  

No significant difference in robustness were found between both our learning rules, besides an advantage for storkey rule for size of networks.


### Image retrieval experiment
![snail](https://user-images.githubusercontent.com/71345328/149384135-23b5377c-1709-4fe0-a41f-f369b80e1569.png) 

https://user-images.githubusercontent.com/71345328/149382681-0d3ce92f-022e-45f1-9c22-ab4785b1d131.mp4

#### Comments
We can see our the retrieval of a binarized picture of a snail via the hebbian learning rule.


## General conclusion
Unfortunately we weren't able to compare both learning rules capacity wise because the values for the storkey rule was lacking. Nevertheless, we saw an advantage for the storkey learning rule for robustness on smaller sized networks. 
