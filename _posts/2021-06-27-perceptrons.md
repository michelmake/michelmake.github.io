---
title: "Artificial Neurons: Perceptrons"
published: true
tag: ml
---

This is the first post in a series on machine learning. This series is my attempt to get more familiar with the topic and is heavily based on the [book by Michael Nielsen](http://neuralnetworksanddeeplearning.com/).

<hr>

A *perceptron* is an artificial neuron. Although in machine learning *Sigmoid
neurons* are more commonly used, it is still useful to have a look at
perceptrons.

A perceptron takes multiple binary inputs and returns a single binary output. A
schematic of a perceptron that takes 3 input signals is shown below:


{:refdef: style="text-align: center;"}
![perceptron](/assets/perceptron.svg)
{: refdef}

Using a set of weights $w_j$ for each input $x_1$, the output of a perceptron is
defined as follows:

$$ \begin{equation}
\text{output} =
\begin{cases}
    0,  & \text{if } \sum_j w_j x_j \leq \alpha\\
    1,  & \text{if } \sum_j w_j x_j \geq \alpha
\end{cases}\label{perceptron}
 \end{equation} $$

 Hence the perceptron has output $1$ if the combined weighted value of the
 inputs  exceeds a given threshold $\alpha$.

 You can think of a perceptrons as simple input based decision makers. A simple
 example  in which you could use a perceptron is to decide you need to wear a
 jacket when you go  out. Given the following, a decision can be made:

1. $x_1:$ are there rainy clouds in the sky?
2. $x_2:$ is it hot?
3. $x_3:$ is it windy?

Based on this **binary** input, a decision can easily be made. If you happen to
not care about getting wet from the rain, a higher weight $w_1$ is given to the
corresponding input $x_1$ variable in Equation \eqref{perceptron}. Modifying
the threshold can significantly affect the decision making.

For example lets say you don't mind rain to much, the weight on the first input
is set to $6$. Additionally, the weights for for the heat and wind are set to
$2$ as you do not like either of them. In this situation, setting the threshold
to $5$ or $7$ would lead to different decision making.

From the above, it shows that a perceptron can be a useful as a decision maker.
Of course, a single  perceptron is not very useful in making complex decisions.
Combining multiple perceptrons in a larger network as shown below could,
however, be used for more sophisticated decisions making.

{:refdef: style="text-align: center;"}
![perceptron](/assets/multiple-perceptrons.svg)
{: refdef}

In the above, the first layer of perceptrons is making a decision based on $5$
input variables. The second layer of perceptrons makes a decision based upon the
output of the first layer of perceptrons. Hence, the second layer of perceptrons
can make decisions on a more abstract sophisticated level. And it even more so,
it holds for the third and final layer in the network. Note, in this network,
the output of a single perceptron is used as input for multiple perceptrons.

To rewrite the conditional behaviour of the perceptron in Equation
\eqref{perceptron} a dotproduct can be used:

$$ \begin{equation}
\text{output} =
\begin{cases}
    0,  & \text{if } \vec{w} \cdot \vec{x} + b \leq 0 \\
    1,  & \text{if } \vec{w} \cdot \vec{x} + b \geq 0
\end{cases}\label{perceptron-simplified}
 \end{equation} $$

In this case, the threshold $\alpha$ is replaced by the soc alled *bias*, $b
\equiv - \alpha$. The bias controls how quickly a perceptron's output will
switch for $0$ to $1$ by simply adding an integer to the combined value of the
weighted inputs. Similarly a weight can do the same for a single input.

In simplified words, the bias determines how quickly the perceptron *fires*. A
very large bias will allow the perceptron to fire easily, while a very negative
bias will do the opposite.

A perceptron can aslo be used to construct computational functions such as an
AND or OR gat. For example:

{:refdef: style="text-align: center;"}
![perceptron](/assets/nand-gate.svg)
{: refdef}

This perceptron is set with weights $-2$ for both inputs and a bias of $3$.
Hence binary inputs yields the following outputs:
<center>
<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-zs3r{background-color:#9b9b9b;border-color:#656565;text-align:center;vertical-align:top}
.tg .tg-mtln{background-color:#c0c0c0;border-color:#656565;text-align:center;vertical-align:top}
</style>
<table class="tg">
<thead>
  <tr>
    <th class="tg-zs3r">$x_1$</th>
    <th class="tg-zs3r">$x_2$</th>
    <th class="tg-zs3r">Perceptron function</th>
    <th class="tg-zs3r">Output</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-mtln">$0$</td>
    <td class="tg-mtln">$0$</td>
    <td class="tg-mtln">$0 * (-2) + 0 * (-2) + 3 = 3$</td>
    <td class="tg-mtln"><span style="font-weight:400;font-style:normal">$1$</span></td>
  </tr>
  <tr>
    <td class="tg-mtln"><span style="font-weight:400;font-style:normal">$1$</span></td>
    <td class="tg-mtln"><span style="font-weight:400;font-style:normal">$0$</span></td>
    <td class="tg-mtln"><span style="font-weight:400;font-style:normal">$1 * (-2) + 0 * (-2) + 3 = 1$</span></td>
    <td class="tg-mtln"><span style="font-weight:400;font-style:normal">$1$</span></td>
  </tr>
  <tr>
    <td class="tg-mtln"><span style="font-weight:400;font-style:normal">$1$</span></td>
    <td class="tg-mtln"><span style="font-weight:400;font-style:normal">$1$</span></td>
    <td class="tg-mtln"><span style="font-weight:400;font-style:normal">$1 * (-2) + 1 * (-2) + 3 = -1$</span></td>
    <td class="tg-mtln"><span style="font-weight:400;font-style:normal">$0$</span></td>
  </tr>
</tbody>
</table>
</center>

This fact gives us the ability to construct any logic function simply using
perceptron with the right weights and biases.


***To be continued***
