# Linear Regression Function

## Model Representation

To establish notation for future use, we'll use $x^{(i)}$ to denote the "input"
variables (living area in this example), also called input features,
and $y^{(i)}$ to denote the "output" or target variable that we are trying
to predict (price). A pair $(x^{(i)},y^{(i)})$ is called a training example,
and the dataset that we'll be using to learn—a list of m training
examples $(x^{(i)},y^{(i)});i=1,...,m$—is called a training set. Note that
the superscript "$(i)$" in the notation is simply an index into the training
set, and has nothing to do with exponentiation. We will also use $X$ to
denote the space of input values, and $Y$ to denote the space of output
values. In this example, $X = Y = ℝ$.

$$
h_θ(x) = θ_0x^{(0)} + θ_1x^{(1)} + ... + θ_ix^{(i)}
$$

$$
x^{(0)} = 1.0
$$

To describe the supervised learning problem slightly more formally, our goal
is, given a training set, to learn a function $h : X → Y$ so that $h(x)$ is a
"good" predictor for the corresponding value of $y$. For historical reasons,
this function $h$ is called a hypothesis. Seen pictorially, the process is
therefore like this:

![H6qTdZmYEeaagxL7xdFKxA_2f0f671110e8f7446bb2b5b2f75a8874_Screenshot-2016-10-23-20 14 58](https://github.com/liangcorp/machine_learning_rust/assets/2737157/dd86f847-2c3b-4efd-ae57-284c1d266376)

When the target variable that we're trying to predict is continuous, such as
in our housing example, we call the learning problem a regression problem.
When $y$ can take on only some discrete values (such as if,
given the living area, we wanted to predict if a dwelling is a house or an
apartment, say), we call it a classification problem.

## Cost Function

We can measure the accuracy of our hypothesis function by using a cost function.
This takes an average difference (actually a fancier version of an average) of
all the results of the hypothesis with inputs from x's and the actual output y's.

This is an example of cost function for 2 features.

$$
J(θ_0, θ_1) = {{1 \over 2m} {\sum_{i=1}^{m}(h_θ(x_i) - y_i)^2}}
$$

## Gradient Descent

When specifically applied to the case of linear regression, a new form
of the gradient descent equation can be derived. We can substitute our
actual cost function and our actual hypothesis function and modify
the equation to :

`repeat until convergence: {`

$$
θ_0 := {θ_0 - α{{1 \over m}{\sum_{i=1}^{m}(h_θ(x^{(i)}) - y^{(i)}) * x_0^{(i)}}}}
$$

$$
θ_1 := {θ_1 - α{{1 \over m}{\sum_{i=1}^{m}(h_θ(x^{(i)}) - y^{(i)}) * x_1^{(i)}}}}
$$

$$
θ_2 := {θ_2 - α{{1 \over m}{\sum_{i=1}^{m}(h_θ(x^{(i)}) - y^{(i)}) * x_2^{(i)}}}}
$$

`}`

in other words:

`repeat until convergence:{`

$$
θ_j := {θ_j - α{{1 \over m}{\sum_{i=1}^{m}(h_θ(x^{(i)}) - y^{(i)}) * x_j^{(i)}}}}
$$

$for j:= 0...n$

`}`

Where $m$ is the size of the training set, $θ_0$ constant that will be
changing simultaneously with $θ_1$ and $x_i,y_i$ are values of the
given training set (data).

Note that we have separated out the two cases for $θ_j$ into separate equations
for $θ_0$ and $θ_1$; and that for $θ_1$ we are multiplying $x_i$ at the end due
to the derivative. The following is a derivation of ${∂ \over ∂θ_j}{J(θ)}$
for a single example:

$$
{∂ \over ∂θ_j}{J(θ)} = {∂ \over ∂θ_j}{1 \over 2}{(h_θ(x) - y)^2}
$$

$$
= 2 * {1 \over 2}(h_θ(x) - y) * {∂ \over ∂θ_j}{(h_θ(x) - y)}
$$

$$
= (h_θ(x) - y) * {∂ \over ∂θ_j}({\sum_{i=0}^{n}}θ_ix_i - y)
$$

$$
= (h_θ(x) - y)x_j
$$

The point of all this is that if we start with a guess for our hypothesis
and then repeatedly apply these gradient descent equations, our hypothesis
will become more and more accurate.

So, this is simply gradient descent on the original cost function $J$.
This method looks at every example in the entire training set on every step,
and is called batch gradient descent. Note that, while gradient descent can be
susceptible to local minima in general, the optimization problem we have posed
here for linear regression has only one global, and no other local, optima;
thus gradient descent always converges (assuming the learning rate $α$ is not
too large) to the global minimum. Indeed, $J$ is a convex quadratic function.
Here is an example of gradient descent as it is run to minimize a quadratic
function.

![Gradient_descent](https://github.com/liangcorp/machine_learning_c/assets/2737157/3b5f0e81-3de5-40e1-8fbf-4e064379a7b4)

The ellipses shown above are the contours of a quadratic function. Also shown
is the trajectory taken by gradient descent, which was initialized at (48,30).
The $x$'s in the figure (joined by straight lines) mark the successive values
of $θ$ that gradient descent went through as it converged to its minimum.

## Normal Equation

Gradient descent gives one way of minimizing $J$. Let’s discuss a second way
of doing so, this time performing the minimization explicitly and without
resorting to an iterative algorithm. In the "Normal Equation" method, we will
minimize $J$ by explicitly taking its derivatives with respect to the $θ_j$'s,
and setting them to zero. This allows us to find the optimum theta without
iteration. The normal equation formula is given below:

$$
θ = (X^TX)^{-1}X^Ty
$$

![dykma6dwEea3qApInhZCFg_333df5f11086fee19c4fb81bc34d5125_Screenshot-2016-11-10-10 06 16](https://github.com/liangcorp/machine_learning_rust/assets/2737157/babdc838-56d3-4e6d-b3f3-a3bef1d8dd9e)

There is no need to do feature scaling with the normal equation.

The following is a comparison of gradient descent and the normal equation:

| Gradient Descent             | Normal Equation                               |
| ---------------------------- | --------------------------------------------- |
| Need to choose alpha         | No need to choose alpha                       |
| Needs many iterations        | No need to iterate                            |
| $O(kn^2)$                    | $O(n^3)$, need to calculate inverse of $X^TX$ |
| Works well when $n$ is large | Slow if $n$ is very large                     |

With the normal equation, computing the inversion has complexity $O(n^3)$.
So if we have a very large number of features, the normal equation will be
slow. In practice, when n exceeds 10,000 it might be a good time to go
from a normal solution to an iterative process.
