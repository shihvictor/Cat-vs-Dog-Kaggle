��
��
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape�"serve*2.2.02v2.2.0-rc4-8-g2b96f3662b8��
�
conv2D_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2D_0/kernel
{
#conv2D_0/kernel/Read/ReadVariableOpReadVariableOpconv2D_0/kernel*&
_output_shapes
: *
dtype0
r
conv2D_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2D_0/bias
k
!conv2D_0/bias/Read/ReadVariableOpReadVariableOpconv2D_0/bias*
_output_shapes
: *
dtype0
l

bn_0/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
bn_0/gamma
e
bn_0/gamma/Read/ReadVariableOpReadVariableOp
bn_0/gamma*
_output_shapes
: *
dtype0
j
	bn_0/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	bn_0/beta
c
bn_0/beta/Read/ReadVariableOpReadVariableOp	bn_0/beta*
_output_shapes
: *
dtype0
x
bn_0/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_namebn_0/moving_mean
q
$bn_0/moving_mean/Read/ReadVariableOpReadVariableOpbn_0/moving_mean*
_output_shapes
: *
dtype0
�
bn_0/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_namebn_0/moving_variance
y
(bn_0/moving_variance/Read/ReadVariableOpReadVariableOpbn_0/moving_variance*
_output_shapes
: *
dtype0
�
conv2D_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @* 
shared_nameconv2D_1/kernel
{
#conv2D_1/kernel/Read/ReadVariableOpReadVariableOpconv2D_1/kernel*&
_output_shapes
: @*
dtype0
r
conv2D_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2D_1/bias
k
!conv2D_1/bias/Read/ReadVariableOpReadVariableOpconv2D_1/bias*
_output_shapes
:@*
dtype0
l

bn_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name
bn_1/gamma
e
bn_1/gamma/Read/ReadVariableOpReadVariableOp
bn_1/gamma*
_output_shapes
:@*
dtype0
j
	bn_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name	bn_1/beta
c
bn_1/beta/Read/ReadVariableOpReadVariableOp	bn_1/beta*
_output_shapes
:@*
dtype0
x
bn_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_namebn_1/moving_mean
q
$bn_1/moving_mean/Read/ReadVariableOpReadVariableOpbn_1/moving_mean*
_output_shapes
:@*
dtype0
�
bn_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_namebn_1/moving_variance
y
(bn_1/moving_variance/Read/ReadVariableOpReadVariableOpbn_1/moving_variance*
_output_shapes
:@*
dtype0
�
conv2D_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�* 
shared_nameconv2D_2/kernel
|
#conv2D_2/kernel/Read/ReadVariableOpReadVariableOpconv2D_2/kernel*'
_output_shapes
:@�*
dtype0
s
conv2D_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv2D_2/bias
l
!conv2D_2/bias/Read/ReadVariableOpReadVariableOpconv2D_2/bias*
_output_shapes	
:�*
dtype0
m

bn_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_name
bn_2/gamma
f
bn_2/gamma/Read/ReadVariableOpReadVariableOp
bn_2/gamma*
_output_shapes	
:�*
dtype0
k
	bn_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_name	bn_2/beta
d
bn_2/beta/Read/ReadVariableOpReadVariableOp	bn_2/beta*
_output_shapes	
:�*
dtype0
y
bn_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*!
shared_namebn_2/moving_mean
r
$bn_2/moving_mean/Read/ReadVariableOpReadVariableOpbn_2/moving_mean*
_output_shapes	
:�*
dtype0
�
bn_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_namebn_2/moving_variance
z
(bn_2/moving_variance/Read/ReadVariableOpReadVariableOpbn_2/moving_variance*
_output_shapes	
:�*
dtype0
t
fc_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
�$�*
shared_namefc_0/kernel
m
fc_0/kernel/Read/ReadVariableOpReadVariableOpfc_0/kernel* 
_output_shapes
:
�$�*
dtype0
k
	fc_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_name	fc_0/bias
d
fc_0/bias/Read/ReadVariableOpReadVariableOp	fc_0/bias*
_output_shapes	
:�*
dtype0
s
fc_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*
shared_namefc_1/kernel
l
fc_1/kernel/Read/ReadVariableOpReadVariableOpfc_1/kernel*
_output_shapes
:	�*
dtype0
j
	fc_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	fc_1/bias
c
fc_1/bias/Read/ReadVariableOpReadVariableOp	fc_1/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
�
Adam/conv2D_0/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv2D_0/kernel/m
�
*Adam/conv2D_0/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2D_0/kernel/m*&
_output_shapes
: *
dtype0
�
Adam/conv2D_0/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv2D_0/bias/m
y
(Adam/conv2D_0/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2D_0/bias/m*
_output_shapes
: *
dtype0
z
Adam/bn_0/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameAdam/bn_0/gamma/m
s
%Adam/bn_0/gamma/m/Read/ReadVariableOpReadVariableOpAdam/bn_0/gamma/m*
_output_shapes
: *
dtype0
x
Adam/bn_0/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameAdam/bn_0/beta/m
q
$Adam/bn_0/beta/m/Read/ReadVariableOpReadVariableOpAdam/bn_0/beta/m*
_output_shapes
: *
dtype0
�
Adam/conv2D_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*'
shared_nameAdam/conv2D_1/kernel/m
�
*Adam/conv2D_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2D_1/kernel/m*&
_output_shapes
: @*
dtype0
�
Adam/conv2D_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2D_1/bias/m
y
(Adam/conv2D_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2D_1/bias/m*
_output_shapes
:@*
dtype0
z
Adam/bn_1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameAdam/bn_1/gamma/m
s
%Adam/bn_1/gamma/m/Read/ReadVariableOpReadVariableOpAdam/bn_1/gamma/m*
_output_shapes
:@*
dtype0
x
Adam/bn_1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_nameAdam/bn_1/beta/m
q
$Adam/bn_1/beta/m/Read/ReadVariableOpReadVariableOpAdam/bn_1/beta/m*
_output_shapes
:@*
dtype0
�
Adam/conv2D_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�*'
shared_nameAdam/conv2D_2/kernel/m
�
*Adam/conv2D_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2D_2/kernel/m*'
_output_shapes
:@�*
dtype0
�
Adam/conv2D_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/conv2D_2/bias/m
z
(Adam/conv2D_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2D_2/bias/m*
_output_shapes	
:�*
dtype0
{
Adam/bn_2/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*"
shared_nameAdam/bn_2/gamma/m
t
%Adam/bn_2/gamma/m/Read/ReadVariableOpReadVariableOpAdam/bn_2/gamma/m*
_output_shapes	
:�*
dtype0
y
Adam/bn_2/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*!
shared_nameAdam/bn_2/beta/m
r
$Adam/bn_2/beta/m/Read/ReadVariableOpReadVariableOpAdam/bn_2/beta/m*
_output_shapes	
:�*
dtype0
�
Adam/fc_0/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
�$�*#
shared_nameAdam/fc_0/kernel/m
{
&Adam/fc_0/kernel/m/Read/ReadVariableOpReadVariableOpAdam/fc_0/kernel/m* 
_output_shapes
:
�$�*
dtype0
y
Adam/fc_0/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*!
shared_nameAdam/fc_0/bias/m
r
$Adam/fc_0/bias/m/Read/ReadVariableOpReadVariableOpAdam/fc_0/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/fc_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*#
shared_nameAdam/fc_1/kernel/m
z
&Adam/fc_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/fc_1/kernel/m*
_output_shapes
:	�*
dtype0
x
Adam/fc_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameAdam/fc_1/bias/m
q
$Adam/fc_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/fc_1/bias/m*
_output_shapes
:*
dtype0
�
Adam/conv2D_0/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv2D_0/kernel/v
�
*Adam/conv2D_0/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2D_0/kernel/v*&
_output_shapes
: *
dtype0
�
Adam/conv2D_0/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv2D_0/bias/v
y
(Adam/conv2D_0/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2D_0/bias/v*
_output_shapes
: *
dtype0
z
Adam/bn_0/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameAdam/bn_0/gamma/v
s
%Adam/bn_0/gamma/v/Read/ReadVariableOpReadVariableOpAdam/bn_0/gamma/v*
_output_shapes
: *
dtype0
x
Adam/bn_0/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameAdam/bn_0/beta/v
q
$Adam/bn_0/beta/v/Read/ReadVariableOpReadVariableOpAdam/bn_0/beta/v*
_output_shapes
: *
dtype0
�
Adam/conv2D_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*'
shared_nameAdam/conv2D_1/kernel/v
�
*Adam/conv2D_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2D_1/kernel/v*&
_output_shapes
: @*
dtype0
�
Adam/conv2D_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2D_1/bias/v
y
(Adam/conv2D_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2D_1/bias/v*
_output_shapes
:@*
dtype0
z
Adam/bn_1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameAdam/bn_1/gamma/v
s
%Adam/bn_1/gamma/v/Read/ReadVariableOpReadVariableOpAdam/bn_1/gamma/v*
_output_shapes
:@*
dtype0
x
Adam/bn_1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_nameAdam/bn_1/beta/v
q
$Adam/bn_1/beta/v/Read/ReadVariableOpReadVariableOpAdam/bn_1/beta/v*
_output_shapes
:@*
dtype0
�
Adam/conv2D_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�*'
shared_nameAdam/conv2D_2/kernel/v
�
*Adam/conv2D_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2D_2/kernel/v*'
_output_shapes
:@�*
dtype0
�
Adam/conv2D_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/conv2D_2/bias/v
z
(Adam/conv2D_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2D_2/bias/v*
_output_shapes	
:�*
dtype0
{
Adam/bn_2/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*"
shared_nameAdam/bn_2/gamma/v
t
%Adam/bn_2/gamma/v/Read/ReadVariableOpReadVariableOpAdam/bn_2/gamma/v*
_output_shapes	
:�*
dtype0
y
Adam/bn_2/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*!
shared_nameAdam/bn_2/beta/v
r
$Adam/bn_2/beta/v/Read/ReadVariableOpReadVariableOpAdam/bn_2/beta/v*
_output_shapes	
:�*
dtype0
�
Adam/fc_0/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
�$�*#
shared_nameAdam/fc_0/kernel/v
{
&Adam/fc_0/kernel/v/Read/ReadVariableOpReadVariableOpAdam/fc_0/kernel/v* 
_output_shapes
:
�$�*
dtype0
y
Adam/fc_0/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*!
shared_nameAdam/fc_0/bias/v
r
$Adam/fc_0/bias/v/Read/ReadVariableOpReadVariableOpAdam/fc_0/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/fc_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*#
shared_nameAdam/fc_1/kernel/v
z
&Adam/fc_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/fc_1/kernel/v*
_output_shapes
:	�*
dtype0
x
Adam/fc_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameAdam/fc_1/bias/v
q
$Adam/fc_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/fc_1/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
�r
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�q
value�qB�q B�q
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer_with_weights-3
layer-7
	layer-8

layer-9
layer-10
layer_with_weights-4
layer-11
layer_with_weights-5
layer-12
layer-13
layer-14
layer-15
layer-16
layer_with_weights-6
layer-17
layer-18
layer_with_weights-7
layer-19
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
 
h

kernel
bias
regularization_losses
trainable_variables
	variables
 	keras_api
�
!axis
	"gamma
#beta
$moving_mean
%moving_variance
&regularization_losses
'trainable_variables
(	variables
)	keras_api
R
*regularization_losses
+trainable_variables
,	variables
-	keras_api
R
.regularization_losses
/trainable_variables
0	variables
1	keras_api
R
2regularization_losses
3trainable_variables
4	variables
5	keras_api
h

6kernel
7bias
8regularization_losses
9trainable_variables
:	variables
;	keras_api
�
<axis
	=gamma
>beta
?moving_mean
@moving_variance
Aregularization_losses
Btrainable_variables
C	variables
D	keras_api
R
Eregularization_losses
Ftrainable_variables
G	variables
H	keras_api
R
Iregularization_losses
Jtrainable_variables
K	variables
L	keras_api
R
Mregularization_losses
Ntrainable_variables
O	variables
P	keras_api
h

Qkernel
Rbias
Sregularization_losses
Ttrainable_variables
U	variables
V	keras_api
�
Waxis
	Xgamma
Ybeta
Zmoving_mean
[moving_variance
\regularization_losses
]trainable_variables
^	variables
_	keras_api
R
`regularization_losses
atrainable_variables
b	variables
c	keras_api
R
dregularization_losses
etrainable_variables
f	variables
g	keras_api
R
hregularization_losses
itrainable_variables
j	variables
k	keras_api
R
lregularization_losses
mtrainable_variables
n	variables
o	keras_api
h

pkernel
qbias
rregularization_losses
strainable_variables
t	variables
u	keras_api
R
vregularization_losses
wtrainable_variables
x	variables
y	keras_api
h

zkernel
{bias
|regularization_losses
}trainable_variables
~	variables
	keras_api
�
	�iter
�beta_1
�beta_2

�decay
�learning_ratem�m�"m�#m�6m�7m�=m�>m�Qm�Rm�Xm�Ym�pm�qm�zm�{m�v�v�"v�#v�6v�7v�=v�>v�Qv�Rv�Xv�Yv�pv�qv�zv�{v�
 
v
0
1
"2
#3
64
75
=6
>7
Q8
R9
X10
Y11
p12
q13
z14
{15
�
0
1
"2
#3
$4
%5
66
77
=8
>9
?10
@11
Q12
R13
X14
Y15
Z16
[17
p18
q19
z20
{21
�
regularization_losses
�non_trainable_variables
�metrics
trainable_variables
�layers
�layer_metrics
 �layer_regularization_losses
	variables
 
[Y
VARIABLE_VALUEconv2D_0/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2D_0/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
�
�non_trainable_variables
regularization_losses
�metrics
trainable_variables
�layers
�layer_metrics
 �layer_regularization_losses
	variables
 
US
VARIABLE_VALUE
bn_0/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUE	bn_0/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEbn_0/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUEbn_0/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

"0
#1

"0
#1
$2
%3
�
�non_trainable_variables
&regularization_losses
�metrics
'trainable_variables
�layers
�layer_metrics
 �layer_regularization_losses
(	variables
 
 
 
�
�non_trainable_variables
*regularization_losses
�metrics
+trainable_variables
�layers
�layer_metrics
 �layer_regularization_losses
,	variables
 
 
 
�
�non_trainable_variables
.regularization_losses
�metrics
/trainable_variables
�layers
�layer_metrics
 �layer_regularization_losses
0	variables
 
 
 
�
�non_trainable_variables
2regularization_losses
�metrics
3trainable_variables
�layers
�layer_metrics
 �layer_regularization_losses
4	variables
[Y
VARIABLE_VALUEconv2D_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2D_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

60
71

60
71
�
�non_trainable_variables
8regularization_losses
�metrics
9trainable_variables
�layers
�layer_metrics
 �layer_regularization_losses
:	variables
 
US
VARIABLE_VALUE
bn_1/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUE	bn_1/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEbn_1/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUEbn_1/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

=0
>1

=0
>1
?2
@3
�
�non_trainable_variables
Aregularization_losses
�metrics
Btrainable_variables
�layers
�layer_metrics
 �layer_regularization_losses
C	variables
 
 
 
�
�non_trainable_variables
Eregularization_losses
�metrics
Ftrainable_variables
�layers
�layer_metrics
 �layer_regularization_losses
G	variables
 
 
 
�
�non_trainable_variables
Iregularization_losses
�metrics
Jtrainable_variables
�layers
�layer_metrics
 �layer_regularization_losses
K	variables
 
 
 
�
�non_trainable_variables
Mregularization_losses
�metrics
Ntrainable_variables
�layers
�layer_metrics
 �layer_regularization_losses
O	variables
[Y
VARIABLE_VALUEconv2D_2/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2D_2/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

Q0
R1

Q0
R1
�
�non_trainable_variables
Sregularization_losses
�metrics
Ttrainable_variables
�layers
�layer_metrics
 �layer_regularization_losses
U	variables
 
US
VARIABLE_VALUE
bn_2/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUE	bn_2/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEbn_2/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUEbn_2/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

X0
Y1

X0
Y1
Z2
[3
�
�non_trainable_variables
\regularization_losses
�metrics
]trainable_variables
�layers
�layer_metrics
 �layer_regularization_losses
^	variables
 
 
 
�
�non_trainable_variables
`regularization_losses
�metrics
atrainable_variables
�layers
�layer_metrics
 �layer_regularization_losses
b	variables
 
 
 
�
�non_trainable_variables
dregularization_losses
�metrics
etrainable_variables
�layers
�layer_metrics
 �layer_regularization_losses
f	variables
 
 
 
�
�non_trainable_variables
hregularization_losses
�metrics
itrainable_variables
�layers
�layer_metrics
 �layer_regularization_losses
j	variables
 
 
 
�
�non_trainable_variables
lregularization_losses
�metrics
mtrainable_variables
�layers
�layer_metrics
 �layer_regularization_losses
n	variables
WU
VARIABLE_VALUEfc_0/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUE	fc_0/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
 

p0
q1

p0
q1
�
�non_trainable_variables
rregularization_losses
�metrics
strainable_variables
�layers
�layer_metrics
 �layer_regularization_losses
t	variables
 
 
 
�
�non_trainable_variables
vregularization_losses
�metrics
wtrainable_variables
�layers
�layer_metrics
 �layer_regularization_losses
x	variables
WU
VARIABLE_VALUEfc_1/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUE	fc_1/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE
 

z0
{1

z0
{1
�
�non_trainable_variables
|regularization_losses
�metrics
}trainable_variables
�layers
�layer_metrics
 �layer_regularization_losses
~	variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
*
$0
%1
?2
@3
Z4
[5

�0
�1
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
 
 
 
 
 
 
 

$0
%1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

?0
@1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

Z0
[1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

�total

�count
�	variables
�	keras_api
I

�total

�count
�
_fn_kwargs
�	variables
�	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

�0
�1

�	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

�0
�1

�	variables
~|
VARIABLE_VALUEAdam/conv2D_0/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2D_0/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/bn_0/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/bn_0/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2D_1/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2D_1/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/bn_1/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/bn_1/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2D_2/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2D_2/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/bn_2/gamma/mQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/bn_2/beta/mPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/fc_0/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/fc_0/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/fc_1/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/fc_1/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2D_0/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2D_0/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/bn_0/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/bn_0/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2D_1/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2D_1/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/bn_1/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/bn_1/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2D_2/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2D_2/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/bn_2/gamma/vQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/bn_2/beta/vPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/fc_0/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/fc_0/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/fc_1/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/fc_1/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�
serving_default_input_1Placeholder*/
_output_shapes
:���������@@*
dtype0*$
shape:���������@@
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv2D_0/kernelconv2D_0/bias
bn_0/gamma	bn_0/betabn_0/moving_meanbn_0/moving_varianceconv2D_1/kernelconv2D_1/bias
bn_1/gamma	bn_1/betabn_1/moving_meanbn_1/moving_varianceconv2D_2/kernelconv2D_2/bias
bn_2/gamma	bn_2/betabn_2/moving_meanbn_2/moving_variancefc_0/kernel	fc_0/biasfc_1/kernel	fc_1/bias*"
Tin
2*
Tout
2*'
_output_shapes
:���������*8
_read_only_resource_inputs
	
**
config_proto

CPU

GPU 2J 8*,
f'R%
#__inference_signature_wrapper_35862
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#conv2D_0/kernel/Read/ReadVariableOp!conv2D_0/bias/Read/ReadVariableOpbn_0/gamma/Read/ReadVariableOpbn_0/beta/Read/ReadVariableOp$bn_0/moving_mean/Read/ReadVariableOp(bn_0/moving_variance/Read/ReadVariableOp#conv2D_1/kernel/Read/ReadVariableOp!conv2D_1/bias/Read/ReadVariableOpbn_1/gamma/Read/ReadVariableOpbn_1/beta/Read/ReadVariableOp$bn_1/moving_mean/Read/ReadVariableOp(bn_1/moving_variance/Read/ReadVariableOp#conv2D_2/kernel/Read/ReadVariableOp!conv2D_2/bias/Read/ReadVariableOpbn_2/gamma/Read/ReadVariableOpbn_2/beta/Read/ReadVariableOp$bn_2/moving_mean/Read/ReadVariableOp(bn_2/moving_variance/Read/ReadVariableOpfc_0/kernel/Read/ReadVariableOpfc_0/bias/Read/ReadVariableOpfc_1/kernel/Read/ReadVariableOpfc_1/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp*Adam/conv2D_0/kernel/m/Read/ReadVariableOp(Adam/conv2D_0/bias/m/Read/ReadVariableOp%Adam/bn_0/gamma/m/Read/ReadVariableOp$Adam/bn_0/beta/m/Read/ReadVariableOp*Adam/conv2D_1/kernel/m/Read/ReadVariableOp(Adam/conv2D_1/bias/m/Read/ReadVariableOp%Adam/bn_1/gamma/m/Read/ReadVariableOp$Adam/bn_1/beta/m/Read/ReadVariableOp*Adam/conv2D_2/kernel/m/Read/ReadVariableOp(Adam/conv2D_2/bias/m/Read/ReadVariableOp%Adam/bn_2/gamma/m/Read/ReadVariableOp$Adam/bn_2/beta/m/Read/ReadVariableOp&Adam/fc_0/kernel/m/Read/ReadVariableOp$Adam/fc_0/bias/m/Read/ReadVariableOp&Adam/fc_1/kernel/m/Read/ReadVariableOp$Adam/fc_1/bias/m/Read/ReadVariableOp*Adam/conv2D_0/kernel/v/Read/ReadVariableOp(Adam/conv2D_0/bias/v/Read/ReadVariableOp%Adam/bn_0/gamma/v/Read/ReadVariableOp$Adam/bn_0/beta/v/Read/ReadVariableOp*Adam/conv2D_1/kernel/v/Read/ReadVariableOp(Adam/conv2D_1/bias/v/Read/ReadVariableOp%Adam/bn_1/gamma/v/Read/ReadVariableOp$Adam/bn_1/beta/v/Read/ReadVariableOp*Adam/conv2D_2/kernel/v/Read/ReadVariableOp(Adam/conv2D_2/bias/v/Read/ReadVariableOp%Adam/bn_2/gamma/v/Read/ReadVariableOp$Adam/bn_2/beta/v/Read/ReadVariableOp&Adam/fc_0/kernel/v/Read/ReadVariableOp$Adam/fc_0/bias/v/Read/ReadVariableOp&Adam/fc_1/kernel/v/Read/ReadVariableOp$Adam/fc_1/bias/v/Read/ReadVariableOpConst*L
TinE
C2A	*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*'
f"R 
__inference__traced_save_37098
�

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2D_0/kernelconv2D_0/bias
bn_0/gamma	bn_0/betabn_0/moving_meanbn_0/moving_varianceconv2D_1/kernelconv2D_1/bias
bn_1/gamma	bn_1/betabn_1/moving_meanbn_1/moving_varianceconv2D_2/kernelconv2D_2/bias
bn_2/gamma	bn_2/betabn_2/moving_meanbn_2/moving_variancefc_0/kernel	fc_0/biasfc_1/kernel	fc_1/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/conv2D_0/kernel/mAdam/conv2D_0/bias/mAdam/bn_0/gamma/mAdam/bn_0/beta/mAdam/conv2D_1/kernel/mAdam/conv2D_1/bias/mAdam/bn_1/gamma/mAdam/bn_1/beta/mAdam/conv2D_2/kernel/mAdam/conv2D_2/bias/mAdam/bn_2/gamma/mAdam/bn_2/beta/mAdam/fc_0/kernel/mAdam/fc_0/bias/mAdam/fc_1/kernel/mAdam/fc_1/bias/mAdam/conv2D_0/kernel/vAdam/conv2D_0/bias/vAdam/bn_0/gamma/vAdam/bn_0/beta/vAdam/conv2D_1/kernel/vAdam/conv2D_1/bias/vAdam/bn_1/gamma/vAdam/bn_1/beta/vAdam/conv2D_2/kernel/vAdam/conv2D_2/bias/vAdam/bn_2/gamma/vAdam/bn_2/beta/vAdam/fc_0/kernel/vAdam/fc_0/bias/vAdam/fc_1/kernel/vAdam/fc_1/bias/v*K
TinD
B2@*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8**
f%R#
!__inference__traced_restore_37299��
�
�
?__inference_bn_1_layer_call_and_return_conditional_losses_35155

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@:::::W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ڀ
�
__inference__traced_save_37098
file_prefix.
*savev2_conv2d_0_kernel_read_readvariableop,
(savev2_conv2d_0_bias_read_readvariableop)
%savev2_bn_0_gamma_read_readvariableop(
$savev2_bn_0_beta_read_readvariableop/
+savev2_bn_0_moving_mean_read_readvariableop3
/savev2_bn_0_moving_variance_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop)
%savev2_bn_1_gamma_read_readvariableop(
$savev2_bn_1_beta_read_readvariableop/
+savev2_bn_1_moving_mean_read_readvariableop3
/savev2_bn_1_moving_variance_read_readvariableop.
*savev2_conv2d_2_kernel_read_readvariableop,
(savev2_conv2d_2_bias_read_readvariableop)
%savev2_bn_2_gamma_read_readvariableop(
$savev2_bn_2_beta_read_readvariableop/
+savev2_bn_2_moving_mean_read_readvariableop3
/savev2_bn_2_moving_variance_read_readvariableop*
&savev2_fc_0_kernel_read_readvariableop(
$savev2_fc_0_bias_read_readvariableop*
&savev2_fc_1_kernel_read_readvariableop(
$savev2_fc_1_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop5
1savev2_adam_conv2d_0_kernel_m_read_readvariableop3
/savev2_adam_conv2d_0_bias_m_read_readvariableop0
,savev2_adam_bn_0_gamma_m_read_readvariableop/
+savev2_adam_bn_0_beta_m_read_readvariableop5
1savev2_adam_conv2d_1_kernel_m_read_readvariableop3
/savev2_adam_conv2d_1_bias_m_read_readvariableop0
,savev2_adam_bn_1_gamma_m_read_readvariableop/
+savev2_adam_bn_1_beta_m_read_readvariableop5
1savev2_adam_conv2d_2_kernel_m_read_readvariableop3
/savev2_adam_conv2d_2_bias_m_read_readvariableop0
,savev2_adam_bn_2_gamma_m_read_readvariableop/
+savev2_adam_bn_2_beta_m_read_readvariableop1
-savev2_adam_fc_0_kernel_m_read_readvariableop/
+savev2_adam_fc_0_bias_m_read_readvariableop1
-savev2_adam_fc_1_kernel_m_read_readvariableop/
+savev2_adam_fc_1_bias_m_read_readvariableop5
1savev2_adam_conv2d_0_kernel_v_read_readvariableop3
/savev2_adam_conv2d_0_bias_v_read_readvariableop0
,savev2_adam_bn_0_gamma_v_read_readvariableop/
+savev2_adam_bn_0_beta_v_read_readvariableop5
1savev2_adam_conv2d_1_kernel_v_read_readvariableop3
/savev2_adam_conv2d_1_bias_v_read_readvariableop0
,savev2_adam_bn_1_gamma_v_read_readvariableop/
+savev2_adam_bn_1_beta_v_read_readvariableop5
1savev2_adam_conv2d_2_kernel_v_read_readvariableop3
/savev2_adam_conv2d_2_bias_v_read_readvariableop0
,savev2_adam_bn_2_gamma_v_read_readvariableop/
+savev2_adam_bn_2_beta_v_read_readvariableop1
-savev2_adam_fc_0_kernel_v_read_readvariableop/
+savev2_adam_fc_0_bias_v_read_readvariableop1
-savev2_adam_fc_1_kernel_v_read_readvariableop/
+savev2_adam_fc_1_bias_v_read_readvariableop
savev2_1_const

identity_1��MergeV2Checkpoints�SaveV2�SaveV2_1�
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const�
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_d0391b8d7a7243d794b72e6828a72172/part2	
Const_1�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�#
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:?*
dtype0*�"
value�"B�"?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:?*
dtype0*�
value�B�?B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_conv2d_0_kernel_read_readvariableop(savev2_conv2d_0_bias_read_readvariableop%savev2_bn_0_gamma_read_readvariableop$savev2_bn_0_beta_read_readvariableop+savev2_bn_0_moving_mean_read_readvariableop/savev2_bn_0_moving_variance_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop%savev2_bn_1_gamma_read_readvariableop$savev2_bn_1_beta_read_readvariableop+savev2_bn_1_moving_mean_read_readvariableop/savev2_bn_1_moving_variance_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop%savev2_bn_2_gamma_read_readvariableop$savev2_bn_2_beta_read_readvariableop+savev2_bn_2_moving_mean_read_readvariableop/savev2_bn_2_moving_variance_read_readvariableop&savev2_fc_0_kernel_read_readvariableop$savev2_fc_0_bias_read_readvariableop&savev2_fc_1_kernel_read_readvariableop$savev2_fc_1_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop1savev2_adam_conv2d_0_kernel_m_read_readvariableop/savev2_adam_conv2d_0_bias_m_read_readvariableop,savev2_adam_bn_0_gamma_m_read_readvariableop+savev2_adam_bn_0_beta_m_read_readvariableop1savev2_adam_conv2d_1_kernel_m_read_readvariableop/savev2_adam_conv2d_1_bias_m_read_readvariableop,savev2_adam_bn_1_gamma_m_read_readvariableop+savev2_adam_bn_1_beta_m_read_readvariableop1savev2_adam_conv2d_2_kernel_m_read_readvariableop/savev2_adam_conv2d_2_bias_m_read_readvariableop,savev2_adam_bn_2_gamma_m_read_readvariableop+savev2_adam_bn_2_beta_m_read_readvariableop-savev2_adam_fc_0_kernel_m_read_readvariableop+savev2_adam_fc_0_bias_m_read_readvariableop-savev2_adam_fc_1_kernel_m_read_readvariableop+savev2_adam_fc_1_bias_m_read_readvariableop1savev2_adam_conv2d_0_kernel_v_read_readvariableop/savev2_adam_conv2d_0_bias_v_read_readvariableop,savev2_adam_bn_0_gamma_v_read_readvariableop+savev2_adam_bn_0_beta_v_read_readvariableop1savev2_adam_conv2d_1_kernel_v_read_readvariableop/savev2_adam_conv2d_1_bias_v_read_readvariableop,savev2_adam_bn_1_gamma_v_read_readvariableop+savev2_adam_bn_1_beta_v_read_readvariableop1savev2_adam_conv2d_2_kernel_v_read_readvariableop/savev2_adam_conv2d_2_bias_v_read_readvariableop,savev2_adam_bn_2_gamma_v_read_readvariableop+savev2_adam_bn_2_beta_v_read_readvariableop-savev2_adam_fc_0_kernel_v_read_readvariableop+savev2_adam_fc_0_bias_v_read_readvariableop-savev2_adam_fc_1_kernel_v_read_readvariableop+savev2_adam_fc_1_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *M
dtypesC
A2?	2
SaveV2�
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shard�
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1�
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names�
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices�
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity�

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*�
_input_shapes�
�: : : : : : : : @:@:@:@:@:@:@�:�:�:�:�:�:
�$�:�:	�:: : : : : : : : : : : : : : @:@:@:@:@�:�:�:�:
�$�:�:	�:: : : : : @:@:@:@:@�:�:�:�:
�$�:�:	�:: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@: 	

_output_shapes
:@: 


_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:-)
'
_output_shapes
:@�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:&"
 
_output_shapes
:
�$�:!

_output_shapes	
:�:%!

_output_shapes
:	�: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :, (
&
_output_shapes
: : !

_output_shapes
: : "

_output_shapes
: : #

_output_shapes
: :,$(
&
_output_shapes
: @: %

_output_shapes
:@: &

_output_shapes
:@: '

_output_shapes
:@:-()
'
_output_shapes
:@�:!)

_output_shapes	
:�:!*

_output_shapes	
:�:!+

_output_shapes	
:�:&,"
 
_output_shapes
:
�$�:!-

_output_shapes	
:�:%.!

_output_shapes
:	�: /

_output_shapes
::,0(
&
_output_shapes
: : 1

_output_shapes
: : 2

_output_shapes
: : 3

_output_shapes
: :,4(
&
_output_shapes
: @: 5

_output_shapes
:@: 6

_output_shapes
:@: 7

_output_shapes
:@:-8)
'
_output_shapes
:@�:!9

_output_shapes	
:�:!:

_output_shapes	
:�:!;

_output_shapes	
:�:&<"
 
_output_shapes
:
�$�:!=

_output_shapes	
:�:%>!

_output_shapes
:	�: ?

_output_shapes
::@

_output_shapes
: 
�
c
D__inference_dropout_3_layer_call_and_return_conditional_losses_35421

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:����������2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
$__inference_bn_0_layer_call_fn_36294

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:���������>> *&
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*H
fCRA
?__inference_bn_0_layer_call_and_return_conditional_losses_350222
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������>> 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������>> ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������>> 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
׉
�
!__inference__traced_restore_37299
file_prefix$
 assignvariableop_conv2d_0_kernel$
 assignvariableop_1_conv2d_0_bias!
assignvariableop_2_bn_0_gamma 
assignvariableop_3_bn_0_beta'
#assignvariableop_4_bn_0_moving_mean+
'assignvariableop_5_bn_0_moving_variance&
"assignvariableop_6_conv2d_1_kernel$
 assignvariableop_7_conv2d_1_bias!
assignvariableop_8_bn_1_gamma 
assignvariableop_9_bn_1_beta(
$assignvariableop_10_bn_1_moving_mean,
(assignvariableop_11_bn_1_moving_variance'
#assignvariableop_12_conv2d_2_kernel%
!assignvariableop_13_conv2d_2_bias"
assignvariableop_14_bn_2_gamma!
assignvariableop_15_bn_2_beta(
$assignvariableop_16_bn_2_moving_mean,
(assignvariableop_17_bn_2_moving_variance#
assignvariableop_18_fc_0_kernel!
assignvariableop_19_fc_0_bias#
assignvariableop_20_fc_1_kernel!
assignvariableop_21_fc_1_bias!
assignvariableop_22_adam_iter#
assignvariableop_23_adam_beta_1#
assignvariableop_24_adam_beta_2"
assignvariableop_25_adam_decay*
&assignvariableop_26_adam_learning_rate
assignvariableop_27_total
assignvariableop_28_count
assignvariableop_29_total_1
assignvariableop_30_count_1.
*assignvariableop_31_adam_conv2d_0_kernel_m,
(assignvariableop_32_adam_conv2d_0_bias_m)
%assignvariableop_33_adam_bn_0_gamma_m(
$assignvariableop_34_adam_bn_0_beta_m.
*assignvariableop_35_adam_conv2d_1_kernel_m,
(assignvariableop_36_adam_conv2d_1_bias_m)
%assignvariableop_37_adam_bn_1_gamma_m(
$assignvariableop_38_adam_bn_1_beta_m.
*assignvariableop_39_adam_conv2d_2_kernel_m,
(assignvariableop_40_adam_conv2d_2_bias_m)
%assignvariableop_41_adam_bn_2_gamma_m(
$assignvariableop_42_adam_bn_2_beta_m*
&assignvariableop_43_adam_fc_0_kernel_m(
$assignvariableop_44_adam_fc_0_bias_m*
&assignvariableop_45_adam_fc_1_kernel_m(
$assignvariableop_46_adam_fc_1_bias_m.
*assignvariableop_47_adam_conv2d_0_kernel_v,
(assignvariableop_48_adam_conv2d_0_bias_v)
%assignvariableop_49_adam_bn_0_gamma_v(
$assignvariableop_50_adam_bn_0_beta_v.
*assignvariableop_51_adam_conv2d_1_kernel_v,
(assignvariableop_52_adam_conv2d_1_bias_v)
%assignvariableop_53_adam_bn_1_gamma_v(
$assignvariableop_54_adam_bn_1_beta_v.
*assignvariableop_55_adam_conv2d_2_kernel_v,
(assignvariableop_56_adam_conv2d_2_bias_v)
%assignvariableop_57_adam_bn_2_gamma_v(
$assignvariableop_58_adam_bn_2_beta_v*
&assignvariableop_59_adam_fc_0_kernel_v(
$assignvariableop_60_adam_fc_0_bias_v*
&assignvariableop_61_adam_fc_1_kernel_v(
$assignvariableop_62_adam_fc_1_bias_v
identity_64��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_57�AssignVariableOp_58�AssignVariableOp_59�AssignVariableOp_6�AssignVariableOp_60�AssignVariableOp_61�AssignVariableOp_62�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�	RestoreV2�RestoreV2_1�#
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:?*
dtype0*�"
value�"B�"?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:?*
dtype0*�
value�B�?B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*M
dtypesC
A2?	2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOp assignvariableop_conv2d_0_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv2d_0_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOpassignvariableop_2_bn_0_gammaIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOpassignvariableop_3_bn_0_betaIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp#assignvariableop_4_bn_0_moving_meanIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOp'assignvariableop_5_bn_0_moving_varianceIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv2d_1_kernelIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv2d_1_biasIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOpassignvariableop_8_bn_1_gammaIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOpassignvariableop_9_bn_1_betaIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp$assignvariableop_10_bn_1_moving_meanIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOp(assignvariableop_11_bn_1_moving_varianceIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp#assignvariableop_12_conv2d_2_kernelIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOp!assignvariableop_13_conv2d_2_biasIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOpassignvariableop_14_bn_2_gammaIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOpassignvariableop_15_bn_2_betaIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp$assignvariableop_16_bn_2_moving_meanIdentity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOp(assignvariableop_17_bn_2_moving_varianceIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOpassignvariableop_18_fc_0_kernelIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOpassignvariableop_19_fc_0_biasIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOpassignvariableop_20_fc_1_kernelIdentity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOpassignvariableop_21_fc_1_biasIdentity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0	*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOpassignvariableop_22_adam_iterIdentity_22:output:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_22_
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOpassignvariableop_23_adam_beta_1Identity_23:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_23_
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOpassignvariableop_24_adam_beta_2Identity_24:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_24_
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOpassignvariableop_25_adam_decayIdentity_25:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_25_
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:2
Identity_26�
AssignVariableOp_26AssignVariableOp&assignvariableop_26_adam_learning_rateIdentity_26:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_26_
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:2
Identity_27�
AssignVariableOp_27AssignVariableOpassignvariableop_27_totalIdentity_27:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_27_
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:2
Identity_28�
AssignVariableOp_28AssignVariableOpassignvariableop_28_countIdentity_28:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_28_
Identity_29IdentityRestoreV2:tensors:29*
T0*
_output_shapes
:2
Identity_29�
AssignVariableOp_29AssignVariableOpassignvariableop_29_total_1Identity_29:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_29_
Identity_30IdentityRestoreV2:tensors:30*
T0*
_output_shapes
:2
Identity_30�
AssignVariableOp_30AssignVariableOpassignvariableop_30_count_1Identity_30:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_30_
Identity_31IdentityRestoreV2:tensors:31*
T0*
_output_shapes
:2
Identity_31�
AssignVariableOp_31AssignVariableOp*assignvariableop_31_adam_conv2d_0_kernel_mIdentity_31:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_31_
Identity_32IdentityRestoreV2:tensors:32*
T0*
_output_shapes
:2
Identity_32�
AssignVariableOp_32AssignVariableOp(assignvariableop_32_adam_conv2d_0_bias_mIdentity_32:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_32_
Identity_33IdentityRestoreV2:tensors:33*
T0*
_output_shapes
:2
Identity_33�
AssignVariableOp_33AssignVariableOp%assignvariableop_33_adam_bn_0_gamma_mIdentity_33:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_33_
Identity_34IdentityRestoreV2:tensors:34*
T0*
_output_shapes
:2
Identity_34�
AssignVariableOp_34AssignVariableOp$assignvariableop_34_adam_bn_0_beta_mIdentity_34:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_34_
Identity_35IdentityRestoreV2:tensors:35*
T0*
_output_shapes
:2
Identity_35�
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_conv2d_1_kernel_mIdentity_35:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_35_
Identity_36IdentityRestoreV2:tensors:36*
T0*
_output_shapes
:2
Identity_36�
AssignVariableOp_36AssignVariableOp(assignvariableop_36_adam_conv2d_1_bias_mIdentity_36:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_36_
Identity_37IdentityRestoreV2:tensors:37*
T0*
_output_shapes
:2
Identity_37�
AssignVariableOp_37AssignVariableOp%assignvariableop_37_adam_bn_1_gamma_mIdentity_37:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_37_
Identity_38IdentityRestoreV2:tensors:38*
T0*
_output_shapes
:2
Identity_38�
AssignVariableOp_38AssignVariableOp$assignvariableop_38_adam_bn_1_beta_mIdentity_38:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_38_
Identity_39IdentityRestoreV2:tensors:39*
T0*
_output_shapes
:2
Identity_39�
AssignVariableOp_39AssignVariableOp*assignvariableop_39_adam_conv2d_2_kernel_mIdentity_39:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_39_
Identity_40IdentityRestoreV2:tensors:40*
T0*
_output_shapes
:2
Identity_40�
AssignVariableOp_40AssignVariableOp(assignvariableop_40_adam_conv2d_2_bias_mIdentity_40:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_40_
Identity_41IdentityRestoreV2:tensors:41*
T0*
_output_shapes
:2
Identity_41�
AssignVariableOp_41AssignVariableOp%assignvariableop_41_adam_bn_2_gamma_mIdentity_41:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_41_
Identity_42IdentityRestoreV2:tensors:42*
T0*
_output_shapes
:2
Identity_42�
AssignVariableOp_42AssignVariableOp$assignvariableop_42_adam_bn_2_beta_mIdentity_42:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_42_
Identity_43IdentityRestoreV2:tensors:43*
T0*
_output_shapes
:2
Identity_43�
AssignVariableOp_43AssignVariableOp&assignvariableop_43_adam_fc_0_kernel_mIdentity_43:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_43_
Identity_44IdentityRestoreV2:tensors:44*
T0*
_output_shapes
:2
Identity_44�
AssignVariableOp_44AssignVariableOp$assignvariableop_44_adam_fc_0_bias_mIdentity_44:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_44_
Identity_45IdentityRestoreV2:tensors:45*
T0*
_output_shapes
:2
Identity_45�
AssignVariableOp_45AssignVariableOp&assignvariableop_45_adam_fc_1_kernel_mIdentity_45:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_45_
Identity_46IdentityRestoreV2:tensors:46*
T0*
_output_shapes
:2
Identity_46�
AssignVariableOp_46AssignVariableOp$assignvariableop_46_adam_fc_1_bias_mIdentity_46:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_46_
Identity_47IdentityRestoreV2:tensors:47*
T0*
_output_shapes
:2
Identity_47�
AssignVariableOp_47AssignVariableOp*assignvariableop_47_adam_conv2d_0_kernel_vIdentity_47:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_47_
Identity_48IdentityRestoreV2:tensors:48*
T0*
_output_shapes
:2
Identity_48�
AssignVariableOp_48AssignVariableOp(assignvariableop_48_adam_conv2d_0_bias_vIdentity_48:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_48_
Identity_49IdentityRestoreV2:tensors:49*
T0*
_output_shapes
:2
Identity_49�
AssignVariableOp_49AssignVariableOp%assignvariableop_49_adam_bn_0_gamma_vIdentity_49:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_49_
Identity_50IdentityRestoreV2:tensors:50*
T0*
_output_shapes
:2
Identity_50�
AssignVariableOp_50AssignVariableOp$assignvariableop_50_adam_bn_0_beta_vIdentity_50:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_50_
Identity_51IdentityRestoreV2:tensors:51*
T0*
_output_shapes
:2
Identity_51�
AssignVariableOp_51AssignVariableOp*assignvariableop_51_adam_conv2d_1_kernel_vIdentity_51:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_51_
Identity_52IdentityRestoreV2:tensors:52*
T0*
_output_shapes
:2
Identity_52�
AssignVariableOp_52AssignVariableOp(assignvariableop_52_adam_conv2d_1_bias_vIdentity_52:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_52_
Identity_53IdentityRestoreV2:tensors:53*
T0*
_output_shapes
:2
Identity_53�
AssignVariableOp_53AssignVariableOp%assignvariableop_53_adam_bn_1_gamma_vIdentity_53:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_53_
Identity_54IdentityRestoreV2:tensors:54*
T0*
_output_shapes
:2
Identity_54�
AssignVariableOp_54AssignVariableOp$assignvariableop_54_adam_bn_1_beta_vIdentity_54:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_54_
Identity_55IdentityRestoreV2:tensors:55*
T0*
_output_shapes
:2
Identity_55�
AssignVariableOp_55AssignVariableOp*assignvariableop_55_adam_conv2d_2_kernel_vIdentity_55:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_55_
Identity_56IdentityRestoreV2:tensors:56*
T0*
_output_shapes
:2
Identity_56�
AssignVariableOp_56AssignVariableOp(assignvariableop_56_adam_conv2d_2_bias_vIdentity_56:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_56_
Identity_57IdentityRestoreV2:tensors:57*
T0*
_output_shapes
:2
Identity_57�
AssignVariableOp_57AssignVariableOp%assignvariableop_57_adam_bn_2_gamma_vIdentity_57:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_57_
Identity_58IdentityRestoreV2:tensors:58*
T0*
_output_shapes
:2
Identity_58�
AssignVariableOp_58AssignVariableOp$assignvariableop_58_adam_bn_2_beta_vIdentity_58:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_58_
Identity_59IdentityRestoreV2:tensors:59*
T0*
_output_shapes
:2
Identity_59�
AssignVariableOp_59AssignVariableOp&assignvariableop_59_adam_fc_0_kernel_vIdentity_59:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_59_
Identity_60IdentityRestoreV2:tensors:60*
T0*
_output_shapes
:2
Identity_60�
AssignVariableOp_60AssignVariableOp$assignvariableop_60_adam_fc_0_bias_vIdentity_60:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_60_
Identity_61IdentityRestoreV2:tensors:61*
T0*
_output_shapes
:2
Identity_61�
AssignVariableOp_61AssignVariableOp&assignvariableop_61_adam_fc_1_kernel_vIdentity_61:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_61_
Identity_62IdentityRestoreV2:tensors:62*
T0*
_output_shapes
:2
Identity_62�
AssignVariableOp_62AssignVariableOp$assignvariableop_62_adam_fc_1_bias_vIdentity_62:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_62�
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names�
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices�
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_63Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_63�
Identity_64IdentityIdentity_63:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_64"#
identity_64Identity_64:output:0*�
_input_shapes�
�: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: :/

_output_shapes
: :0

_output_shapes
: :1

_output_shapes
: :2

_output_shapes
: :3

_output_shapes
: :4

_output_shapes
: :5

_output_shapes
: :6

_output_shapes
: :7

_output_shapes
: :8

_output_shapes
: :9

_output_shapes
: ::

_output_shapes
: :;

_output_shapes
: :<

_output_shapes
: :=

_output_shapes
: :>

_output_shapes
: :?

_output_shapes
: 
�
c
D__inference_dropout_1_layer_call_and_return_conditional_losses_35217

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:���������@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:���������@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������@2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:���������@2
dropout/Cast�
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:���������@2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
b
)__inference_dropout_1_layer_call_fn_36600

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*/
_output_shapes
:���������@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_352172
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�$
�
?__inference_bn_1_layer_call_and_return_conditional_losses_36524

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��#AssignMovingAvg/AssignSubVariableOp�%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
E
)__inference_dropout_1_layer_call_fn_36605

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*/
_output_shapes
:���������@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_352222
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�$
�
?__inference_bn_2_layer_call_and_return_conditional_losses_36723

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��#AssignMovingAvg/AssignSubVariableOp�%AssignMovingAvg_1/AssignSubVariableOpu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:�2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:�2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:�2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:�2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:����������::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
?__inference_bn_0_layer_call_and_return_conditional_losses_36268

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������>> : : : : :*
epsilon%o�:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������>> 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������>> :::::W S
/
_output_shapes
:���������>> 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
}
(__inference_conv2D_2_layer_call_fn_34827

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_conv2D_2_layer_call_and_return_conditional_losses_348172
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������@::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
b
D__inference_dropout_2_layer_call_and_return_conditional_losses_36794

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:����������2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:����������2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
b
)__inference_dropout_2_layer_call_fn_36799

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*0
_output_shapes
:����������* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_353502
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
a
E__inference_max_pool_1_layer_call_and_return_conditional_losses_34800

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
2	
MaxPool�
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�Q
�
F__inference_CatDogModel_layer_call_and_return_conditional_losses_35604

inputs
conv2d_0_35540
conv2d_0_35542

bn_0_35545

bn_0_35547

bn_0_35549

bn_0_35551
conv2d_1_35557
conv2d_1_35559

bn_1_35562

bn_1_35564

bn_1_35566

bn_1_35568
conv2d_2_35574
conv2d_2_35576

bn_2_35579

bn_2_35581

bn_2_35583

bn_2_35585

fc_0_35592

fc_0_35594

fc_1_35598

fc_1_35600
identity��bn_0/StatefulPartitionedCall�bn_1/StatefulPartitionedCall�bn_2/StatefulPartitionedCall� conv2D_0/StatefulPartitionedCall� conv2D_1/StatefulPartitionedCall� conv2D_2/StatefulPartitionedCall�dropout/StatefulPartitionedCall�!dropout_1/StatefulPartitionedCall�!dropout_2/StatefulPartitionedCall�!dropout_3/StatefulPartitionedCall�fc_0/StatefulPartitionedCall�fc_1/StatefulPartitionedCall�
 conv2D_0/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_0_35540conv2d_0_35542*
Tin
2*
Tout
2*/
_output_shapes
:���������>> *$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_conv2D_0_layer_call_and_return_conditional_losses_344992"
 conv2D_0/StatefulPartitionedCall�
bn_0/StatefulPartitionedCallStatefulPartitionedCall)conv2D_0/StatefulPartitionedCall:output:0
bn_0_35545
bn_0_35547
bn_0_35549
bn_0_35551*
Tin	
2*
Tout
2*/
_output_shapes
:���������>> *$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*H
fCRA
?__inference_bn_0_layer_call_and_return_conditional_losses_350042
bn_0/StatefulPartitionedCall�
activation/PartitionedCallPartitionedCall%bn_0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:���������>> * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_350632
activation/PartitionedCall�
max_pool_0/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:��������� * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_max_pool_0_layer_call_and_return_conditional_losses_346412
max_pool_0/PartitionedCall�
dropout/StatefulPartitionedCallStatefulPartitionedCall#max_pool_0/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:��������� * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_350842!
dropout/StatefulPartitionedCall�
 conv2D_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0conv2d_1_35557conv2d_1_35559*
Tin
2*
Tout
2*/
_output_shapes
:���������@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_conv2D_1_layer_call_and_return_conditional_losses_346582"
 conv2D_1/StatefulPartitionedCall�
bn_1/StatefulPartitionedCallStatefulPartitionedCall)conv2D_1/StatefulPartitionedCall:output:0
bn_1_35562
bn_1_35564
bn_1_35566
bn_1_35568*
Tin	
2*
Tout
2*/
_output_shapes
:���������@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*H
fCRA
?__inference_bn_1_layer_call_and_return_conditional_losses_351372
bn_1/StatefulPartitionedCall�
activation_1/PartitionedCallPartitionedCall%bn_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:���������@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_351962
activation_1/PartitionedCall�
max_pool_1/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:���������@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_max_pool_1_layer_call_and_return_conditional_losses_348002
max_pool_1/PartitionedCall�
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall#max_pool_1/PartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*/
_output_shapes
:���������@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_352172#
!dropout_1/StatefulPartitionedCall�
 conv2D_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0conv2d_2_35574conv2d_2_35576*
Tin
2*
Tout
2*0
_output_shapes
:����������*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_conv2D_2_layer_call_and_return_conditional_losses_348172"
 conv2D_2/StatefulPartitionedCall�
bn_2/StatefulPartitionedCallStatefulPartitionedCall)conv2D_2/StatefulPartitionedCall:output:0
bn_2_35579
bn_2_35581
bn_2_35583
bn_2_35585*
Tin	
2*
Tout
2*0
_output_shapes
:����������*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*H
fCRA
?__inference_bn_2_layer_call_and_return_conditional_losses_352702
bn_2/StatefulPartitionedCall�
activation_2/PartitionedCallPartitionedCall%bn_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*0
_output_shapes
:����������* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*P
fKRI
G__inference_activation_2_layer_call_and_return_conditional_losses_353292
activation_2/PartitionedCall�
max_pool_2/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*0
_output_shapes
:����������* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_max_pool_2_layer_call_and_return_conditional_losses_349592
max_pool_2/PartitionedCall�
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall#max_pool_2/PartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*0
_output_shapes
:����������* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_353502#
!dropout_2/StatefulPartitionedCall�
flatten/PartitionedCallPartitionedCall*dropout_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:����������$* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_353742
flatten/PartitionedCall�
fc_0/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0
fc_0_35592
fc_0_35594*
Tin
2*
Tout
2*(
_output_shapes
:����������*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*H
fCRA
?__inference_fc_0_layer_call_and_return_conditional_losses_353932
fc_0/StatefulPartitionedCall�
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall%fc_0/StatefulPartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*(
_output_shapes
:����������* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_354212#
!dropout_3/StatefulPartitionedCall�
fc_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_3/StatefulPartitionedCall:output:0
fc_1_35598
fc_1_35600*
Tin
2*
Tout
2*'
_output_shapes
:���������*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*H
fCRA
?__inference_fc_1_layer_call_and_return_conditional_losses_354502
fc_1/StatefulPartitionedCall�
IdentityIdentity%fc_1/StatefulPartitionedCall:output:0^bn_0/StatefulPartitionedCall^bn_1/StatefulPartitionedCall^bn_2/StatefulPartitionedCall!^conv2D_0/StatefulPartitionedCall!^conv2D_1/StatefulPartitionedCall!^conv2D_2/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall^fc_0/StatefulPartitionedCall^fc_1/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapesu
s:���������@@::::::::::::::::::::::2<
bn_0/StatefulPartitionedCallbn_0/StatefulPartitionedCall2<
bn_1/StatefulPartitionedCallbn_1/StatefulPartitionedCall2<
bn_2/StatefulPartitionedCallbn_2/StatefulPartitionedCall2D
 conv2D_0/StatefulPartitionedCall conv2D_0/StatefulPartitionedCall2D
 conv2D_1/StatefulPartitionedCall conv2D_1/StatefulPartitionedCall2D
 conv2D_2/StatefulPartitionedCall conv2D_2/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall2<
fc_0/StatefulPartitionedCallfc_0/StatefulPartitionedCall2<
fc_1/StatefulPartitionedCallfc_1/StatefulPartitionedCall:W S
/
_output_shapes
:���������@@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�Q
�
F__inference_CatDogModel_layer_call_and_return_conditional_losses_35467
input_1
conv2d_0_34969
conv2d_0_34971

bn_0_35049

bn_0_35051

bn_0_35053

bn_0_35055
conv2d_1_35102
conv2d_1_35104

bn_1_35182

bn_1_35184

bn_1_35186

bn_1_35188
conv2d_2_35235
conv2d_2_35237

bn_2_35315

bn_2_35317

bn_2_35319

bn_2_35321

fc_0_35404

fc_0_35406

fc_1_35461

fc_1_35463
identity��bn_0/StatefulPartitionedCall�bn_1/StatefulPartitionedCall�bn_2/StatefulPartitionedCall� conv2D_0/StatefulPartitionedCall� conv2D_1/StatefulPartitionedCall� conv2D_2/StatefulPartitionedCall�dropout/StatefulPartitionedCall�!dropout_1/StatefulPartitionedCall�!dropout_2/StatefulPartitionedCall�!dropout_3/StatefulPartitionedCall�fc_0/StatefulPartitionedCall�fc_1/StatefulPartitionedCall�
 conv2D_0/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_0_34969conv2d_0_34971*
Tin
2*
Tout
2*/
_output_shapes
:���������>> *$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_conv2D_0_layer_call_and_return_conditional_losses_344992"
 conv2D_0/StatefulPartitionedCall�
bn_0/StatefulPartitionedCallStatefulPartitionedCall)conv2D_0/StatefulPartitionedCall:output:0
bn_0_35049
bn_0_35051
bn_0_35053
bn_0_35055*
Tin	
2*
Tout
2*/
_output_shapes
:���������>> *$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*H
fCRA
?__inference_bn_0_layer_call_and_return_conditional_losses_350042
bn_0/StatefulPartitionedCall�
activation/PartitionedCallPartitionedCall%bn_0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:���������>> * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_350632
activation/PartitionedCall�
max_pool_0/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:��������� * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_max_pool_0_layer_call_and_return_conditional_losses_346412
max_pool_0/PartitionedCall�
dropout/StatefulPartitionedCallStatefulPartitionedCall#max_pool_0/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:��������� * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_350842!
dropout/StatefulPartitionedCall�
 conv2D_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0conv2d_1_35102conv2d_1_35104*
Tin
2*
Tout
2*/
_output_shapes
:���������@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_conv2D_1_layer_call_and_return_conditional_losses_346582"
 conv2D_1/StatefulPartitionedCall�
bn_1/StatefulPartitionedCallStatefulPartitionedCall)conv2D_1/StatefulPartitionedCall:output:0
bn_1_35182
bn_1_35184
bn_1_35186
bn_1_35188*
Tin	
2*
Tout
2*/
_output_shapes
:���������@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*H
fCRA
?__inference_bn_1_layer_call_and_return_conditional_losses_351372
bn_1/StatefulPartitionedCall�
activation_1/PartitionedCallPartitionedCall%bn_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:���������@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_351962
activation_1/PartitionedCall�
max_pool_1/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:���������@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_max_pool_1_layer_call_and_return_conditional_losses_348002
max_pool_1/PartitionedCall�
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall#max_pool_1/PartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*/
_output_shapes
:���������@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_352172#
!dropout_1/StatefulPartitionedCall�
 conv2D_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0conv2d_2_35235conv2d_2_35237*
Tin
2*
Tout
2*0
_output_shapes
:����������*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_conv2D_2_layer_call_and_return_conditional_losses_348172"
 conv2D_2/StatefulPartitionedCall�
bn_2/StatefulPartitionedCallStatefulPartitionedCall)conv2D_2/StatefulPartitionedCall:output:0
bn_2_35315
bn_2_35317
bn_2_35319
bn_2_35321*
Tin	
2*
Tout
2*0
_output_shapes
:����������*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*H
fCRA
?__inference_bn_2_layer_call_and_return_conditional_losses_352702
bn_2/StatefulPartitionedCall�
activation_2/PartitionedCallPartitionedCall%bn_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*0
_output_shapes
:����������* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*P
fKRI
G__inference_activation_2_layer_call_and_return_conditional_losses_353292
activation_2/PartitionedCall�
max_pool_2/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*0
_output_shapes
:����������* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_max_pool_2_layer_call_and_return_conditional_losses_349592
max_pool_2/PartitionedCall�
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall#max_pool_2/PartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*0
_output_shapes
:����������* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_353502#
!dropout_2/StatefulPartitionedCall�
flatten/PartitionedCallPartitionedCall*dropout_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:����������$* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_353742
flatten/PartitionedCall�
fc_0/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0
fc_0_35404
fc_0_35406*
Tin
2*
Tout
2*(
_output_shapes
:����������*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*H
fCRA
?__inference_fc_0_layer_call_and_return_conditional_losses_353932
fc_0/StatefulPartitionedCall�
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall%fc_0/StatefulPartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*(
_output_shapes
:����������* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_354212#
!dropout_3/StatefulPartitionedCall�
fc_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_3/StatefulPartitionedCall:output:0
fc_1_35461
fc_1_35463*
Tin
2*
Tout
2*'
_output_shapes
:���������*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*H
fCRA
?__inference_fc_1_layer_call_and_return_conditional_losses_354502
fc_1/StatefulPartitionedCall�
IdentityIdentity%fc_1/StatefulPartitionedCall:output:0^bn_0/StatefulPartitionedCall^bn_1/StatefulPartitionedCall^bn_2/StatefulPartitionedCall!^conv2D_0/StatefulPartitionedCall!^conv2D_1/StatefulPartitionedCall!^conv2D_2/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall^fc_0/StatefulPartitionedCall^fc_1/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapesu
s:���������@@::::::::::::::::::::::2<
bn_0/StatefulPartitionedCallbn_0/StatefulPartitionedCall2<
bn_1/StatefulPartitionedCallbn_1/StatefulPartitionedCall2<
bn_2/StatefulPartitionedCallbn_2/StatefulPartitionedCall2D
 conv2D_0/StatefulPartitionedCall conv2D_0/StatefulPartitionedCall2D
 conv2D_1/StatefulPartitionedCall conv2D_1/StatefulPartitionedCall2D
 conv2D_2/StatefulPartitionedCall conv2D_2/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall2<
fc_0/StatefulPartitionedCallfc_0/StatefulPartitionedCall2<
fc_1/StatefulPartitionedCallfc_1/StatefulPartitionedCall:X T
/
_output_shapes
:���������@@
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
?__inference_bn_0_layer_call_and_return_conditional_losses_36343

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� :::::i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
`
'__inference_dropout_layer_call_fn_36401

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*/
_output_shapes
:��������� * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_350842
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*.
_input_shapes
:��������� 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
?__inference_bn_2_layer_call_and_return_conditional_losses_36741

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:����������:::::X T
0
_output_shapes
:����������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
c
D__inference_dropout_2_layer_call_and_return_conditional_losses_36789

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:����������2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:����������*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:����������2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:����������2
dropout/Cast�
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:����������2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
?__inference_fc_1_layer_call_and_return_conditional_losses_35450

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:::P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�	
�
C__inference_conv2D_1_layer_call_and_return_conditional_losses_34658

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@*
paddingVALID*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@2	
BiasAdd~
IdentityIdentityBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+��������������������������� :::i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
�
$__inference_bn_0_layer_call_fn_36356

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+��������������������������� *$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*H
fCRA
?__inference_bn_0_layer_call_and_return_conditional_losses_345932
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�$
�
?__inference_bn_1_layer_call_and_return_conditional_losses_34752

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��#AssignMovingAvg/AssignSubVariableOp�%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
?__inference_fc_0_layer_call_and_return_conditional_losses_36826

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
�$�*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������$:::P L
(
_output_shapes
:����������$
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
a
B__inference_dropout_layer_call_and_return_conditional_losses_36391

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:��������� 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:��������� *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:��������� 2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:��������� 2
dropout/Cast�
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:��������� 2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*.
_input_shapes
:��������� :W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
c
D__inference_dropout_1_layer_call_and_return_conditional_losses_36590

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:���������@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:���������@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������@2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:���������@2
dropout/Cast�
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:���������@2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
b
)__inference_dropout_3_layer_call_fn_36857

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:����������* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_354212
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*'
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�$
�
?__inference_bn_2_layer_call_and_return_conditional_losses_34911

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��#AssignMovingAvg/AssignSubVariableOp�%AssignMovingAvg_1/AssignSubVariableOpu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:�2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:�2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:�2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:�2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
y
$__inference_fc_0_layer_call_fn_36835

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*(
_output_shapes
:����������*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*H
fCRA
?__inference_fc_0_layer_call_and_return_conditional_losses_353932
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������$::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������$
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
c
G__inference_activation_2_layer_call_and_return_conditional_losses_35329

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:����������2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
+__inference_CatDogModel_layer_call_fn_36158

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*'
_output_shapes
:���������*2
_read_only_resource_inputs
	
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_CatDogModel_layer_call_and_return_conditional_losses_356042
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapesu
s:���������@@::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������@@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
$__inference_bn_1_layer_call_fn_36555

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:���������@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*H
fCRA
?__inference_bn_1_layer_call_and_return_conditional_losses_351372
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
a
B__inference_dropout_layer_call_and_return_conditional_losses_35084

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:��������� 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:��������� *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:��������� 2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:��������� 2
dropout/Cast�
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:��������� 2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*.
_input_shapes
:��������� :W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
+__inference_CatDogModel_layer_call_fn_35651
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*'
_output_shapes
:���������*2
_read_only_resource_inputs
	
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_CatDogModel_layer_call_and_return_conditional_losses_356042
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapesu
s:���������@@::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������@@
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�K
�
F__inference_CatDogModel_layer_call_and_return_conditional_losses_35534
input_1
conv2d_0_35470
conv2d_0_35472

bn_0_35475

bn_0_35477

bn_0_35479

bn_0_35481
conv2d_1_35487
conv2d_1_35489

bn_1_35492

bn_1_35494

bn_1_35496

bn_1_35498
conv2d_2_35504
conv2d_2_35506

bn_2_35509

bn_2_35511

bn_2_35513

bn_2_35515

fc_0_35522

fc_0_35524

fc_1_35528

fc_1_35530
identity��bn_0/StatefulPartitionedCall�bn_1/StatefulPartitionedCall�bn_2/StatefulPartitionedCall� conv2D_0/StatefulPartitionedCall� conv2D_1/StatefulPartitionedCall� conv2D_2/StatefulPartitionedCall�fc_0/StatefulPartitionedCall�fc_1/StatefulPartitionedCall�
 conv2D_0/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_0_35470conv2d_0_35472*
Tin
2*
Tout
2*/
_output_shapes
:���������>> *$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_conv2D_0_layer_call_and_return_conditional_losses_344992"
 conv2D_0/StatefulPartitionedCall�
bn_0/StatefulPartitionedCallStatefulPartitionedCall)conv2D_0/StatefulPartitionedCall:output:0
bn_0_35475
bn_0_35477
bn_0_35479
bn_0_35481*
Tin	
2*
Tout
2*/
_output_shapes
:���������>> *&
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*H
fCRA
?__inference_bn_0_layer_call_and_return_conditional_losses_350222
bn_0/StatefulPartitionedCall�
activation/PartitionedCallPartitionedCall%bn_0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:���������>> * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_350632
activation/PartitionedCall�
max_pool_0/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:��������� * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_max_pool_0_layer_call_and_return_conditional_losses_346412
max_pool_0/PartitionedCall�
dropout/PartitionedCallPartitionedCall#max_pool_0/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:��������� * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_350892
dropout/PartitionedCall�
 conv2D_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0conv2d_1_35487conv2d_1_35489*
Tin
2*
Tout
2*/
_output_shapes
:���������@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_conv2D_1_layer_call_and_return_conditional_losses_346582"
 conv2D_1/StatefulPartitionedCall�
bn_1/StatefulPartitionedCallStatefulPartitionedCall)conv2D_1/StatefulPartitionedCall:output:0
bn_1_35492
bn_1_35494
bn_1_35496
bn_1_35498*
Tin	
2*
Tout
2*/
_output_shapes
:���������@*&
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*H
fCRA
?__inference_bn_1_layer_call_and_return_conditional_losses_351552
bn_1/StatefulPartitionedCall�
activation_1/PartitionedCallPartitionedCall%bn_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:���������@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_351962
activation_1/PartitionedCall�
max_pool_1/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:���������@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_max_pool_1_layer_call_and_return_conditional_losses_348002
max_pool_1/PartitionedCall�
dropout_1/PartitionedCallPartitionedCall#max_pool_1/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:���������@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_352222
dropout_1/PartitionedCall�
 conv2D_2/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0conv2d_2_35504conv2d_2_35506*
Tin
2*
Tout
2*0
_output_shapes
:����������*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_conv2D_2_layer_call_and_return_conditional_losses_348172"
 conv2D_2/StatefulPartitionedCall�
bn_2/StatefulPartitionedCallStatefulPartitionedCall)conv2D_2/StatefulPartitionedCall:output:0
bn_2_35509
bn_2_35511
bn_2_35513
bn_2_35515*
Tin	
2*
Tout
2*0
_output_shapes
:����������*&
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*H
fCRA
?__inference_bn_2_layer_call_and_return_conditional_losses_352882
bn_2/StatefulPartitionedCall�
activation_2/PartitionedCallPartitionedCall%bn_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*0
_output_shapes
:����������* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*P
fKRI
G__inference_activation_2_layer_call_and_return_conditional_losses_353292
activation_2/PartitionedCall�
max_pool_2/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*0
_output_shapes
:����������* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_max_pool_2_layer_call_and_return_conditional_losses_349592
max_pool_2/PartitionedCall�
dropout_2/PartitionedCallPartitionedCall#max_pool_2/PartitionedCall:output:0*
Tin
2*
Tout
2*0
_output_shapes
:����������* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_353552
dropout_2/PartitionedCall�
flatten/PartitionedCallPartitionedCall"dropout_2/PartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:����������$* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_353742
flatten/PartitionedCall�
fc_0/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0
fc_0_35522
fc_0_35524*
Tin
2*
Tout
2*(
_output_shapes
:����������*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*H
fCRA
?__inference_fc_0_layer_call_and_return_conditional_losses_353932
fc_0/StatefulPartitionedCall�
dropout_3/PartitionedCallPartitionedCall%fc_0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:����������* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_354262
dropout_3/PartitionedCall�
fc_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0
fc_1_35528
fc_1_35530*
Tin
2*
Tout
2*'
_output_shapes
:���������*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*H
fCRA
?__inference_fc_1_layer_call_and_return_conditional_losses_354502
fc_1/StatefulPartitionedCall�
IdentityIdentity%fc_1/StatefulPartitionedCall:output:0^bn_0/StatefulPartitionedCall^bn_1/StatefulPartitionedCall^bn_2/StatefulPartitionedCall!^conv2D_0/StatefulPartitionedCall!^conv2D_1/StatefulPartitionedCall!^conv2D_2/StatefulPartitionedCall^fc_0/StatefulPartitionedCall^fc_1/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapesu
s:���������@@::::::::::::::::::::::2<
bn_0/StatefulPartitionedCallbn_0/StatefulPartitionedCall2<
bn_1/StatefulPartitionedCallbn_1/StatefulPartitionedCall2<
bn_2/StatefulPartitionedCallbn_2/StatefulPartitionedCall2D
 conv2D_0/StatefulPartitionedCall conv2D_0/StatefulPartitionedCall2D
 conv2D_1/StatefulPartitionedCall conv2D_1/StatefulPartitionedCall2D
 conv2D_2/StatefulPartitionedCall conv2D_2/StatefulPartitionedCall2<
fc_0/StatefulPartitionedCallfc_0/StatefulPartitionedCall2<
fc_1/StatefulPartitionedCallfc_1/StatefulPartitionedCall:X T
/
_output_shapes
:���������@@
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
b
D__inference_dropout_3_layer_call_and_return_conditional_losses_36852

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:����������2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�d
�

 __inference__wrapped_model_34488
input_17
3catdogmodel_conv2d_0_conv2d_readvariableop_resource8
4catdogmodel_conv2d_0_biasadd_readvariableop_resource,
(catdogmodel_bn_0_readvariableop_resource.
*catdogmodel_bn_0_readvariableop_1_resource=
9catdogmodel_bn_0_fusedbatchnormv3_readvariableop_resource?
;catdogmodel_bn_0_fusedbatchnormv3_readvariableop_1_resource7
3catdogmodel_conv2d_1_conv2d_readvariableop_resource8
4catdogmodel_conv2d_1_biasadd_readvariableop_resource,
(catdogmodel_bn_1_readvariableop_resource.
*catdogmodel_bn_1_readvariableop_1_resource=
9catdogmodel_bn_1_fusedbatchnormv3_readvariableop_resource?
;catdogmodel_bn_1_fusedbatchnormv3_readvariableop_1_resource7
3catdogmodel_conv2d_2_conv2d_readvariableop_resource8
4catdogmodel_conv2d_2_biasadd_readvariableop_resource,
(catdogmodel_bn_2_readvariableop_resource.
*catdogmodel_bn_2_readvariableop_1_resource=
9catdogmodel_bn_2_fusedbatchnormv3_readvariableop_resource?
;catdogmodel_bn_2_fusedbatchnormv3_readvariableop_1_resource3
/catdogmodel_fc_0_matmul_readvariableop_resource4
0catdogmodel_fc_0_biasadd_readvariableop_resource3
/catdogmodel_fc_1_matmul_readvariableop_resource4
0catdogmodel_fc_1_biasadd_readvariableop_resource
identity��
*CatDogModel/conv2D_0/Conv2D/ReadVariableOpReadVariableOp3catdogmodel_conv2d_0_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02,
*CatDogModel/conv2D_0/Conv2D/ReadVariableOp�
CatDogModel/conv2D_0/Conv2DConv2Dinput_12CatDogModel/conv2D_0/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������>> *
paddingVALID*
strides
2
CatDogModel/conv2D_0/Conv2D�
+CatDogModel/conv2D_0/BiasAdd/ReadVariableOpReadVariableOp4catdogmodel_conv2d_0_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+CatDogModel/conv2D_0/BiasAdd/ReadVariableOp�
CatDogModel/conv2D_0/BiasAddBiasAdd$CatDogModel/conv2D_0/Conv2D:output:03CatDogModel/conv2D_0/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������>> 2
CatDogModel/conv2D_0/BiasAdd�
CatDogModel/bn_0/ReadVariableOpReadVariableOp(catdogmodel_bn_0_readvariableop_resource*
_output_shapes
: *
dtype02!
CatDogModel/bn_0/ReadVariableOp�
!CatDogModel/bn_0/ReadVariableOp_1ReadVariableOp*catdogmodel_bn_0_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!CatDogModel/bn_0/ReadVariableOp_1�
0CatDogModel/bn_0/FusedBatchNormV3/ReadVariableOpReadVariableOp9catdogmodel_bn_0_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype022
0CatDogModel/bn_0/FusedBatchNormV3/ReadVariableOp�
2CatDogModel/bn_0/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp;catdogmodel_bn_0_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype024
2CatDogModel/bn_0/FusedBatchNormV3/ReadVariableOp_1�
!CatDogModel/bn_0/FusedBatchNormV3FusedBatchNormV3%CatDogModel/conv2D_0/BiasAdd:output:0'CatDogModel/bn_0/ReadVariableOp:value:0)CatDogModel/bn_0/ReadVariableOp_1:value:08CatDogModel/bn_0/FusedBatchNormV3/ReadVariableOp:value:0:CatDogModel/bn_0/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������>> : : : : :*
epsilon%o�:*
is_training( 2#
!CatDogModel/bn_0/FusedBatchNormV3�
CatDogModel/activation/ReluRelu%CatDogModel/bn_0/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������>> 2
CatDogModel/activation/Relu�
CatDogModel/max_pool_0/MaxPoolMaxPool)CatDogModel/activation/Relu:activations:0*/
_output_shapes
:��������� *
ksize
*
paddingVALID*
strides
2 
CatDogModel/max_pool_0/MaxPool�
CatDogModel/dropout/IdentityIdentity'CatDogModel/max_pool_0/MaxPool:output:0*
T0*/
_output_shapes
:��������� 2
CatDogModel/dropout/Identity�
*CatDogModel/conv2D_1/Conv2D/ReadVariableOpReadVariableOp3catdogmodel_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02,
*CatDogModel/conv2D_1/Conv2D/ReadVariableOp�
CatDogModel/conv2D_1/Conv2DConv2D%CatDogModel/dropout/Identity:output:02CatDogModel/conv2D_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingVALID*
strides
2
CatDogModel/conv2D_1/Conv2D�
+CatDogModel/conv2D_1/BiasAdd/ReadVariableOpReadVariableOp4catdogmodel_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+CatDogModel/conv2D_1/BiasAdd/ReadVariableOp�
CatDogModel/conv2D_1/BiasAddBiasAdd$CatDogModel/conv2D_1/Conv2D:output:03CatDogModel/conv2D_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
CatDogModel/conv2D_1/BiasAdd�
CatDogModel/bn_1/ReadVariableOpReadVariableOp(catdogmodel_bn_1_readvariableop_resource*
_output_shapes
:@*
dtype02!
CatDogModel/bn_1/ReadVariableOp�
!CatDogModel/bn_1/ReadVariableOp_1ReadVariableOp*catdogmodel_bn_1_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!CatDogModel/bn_1/ReadVariableOp_1�
0CatDogModel/bn_1/FusedBatchNormV3/ReadVariableOpReadVariableOp9catdogmodel_bn_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype022
0CatDogModel/bn_1/FusedBatchNormV3/ReadVariableOp�
2CatDogModel/bn_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp;catdogmodel_bn_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype024
2CatDogModel/bn_1/FusedBatchNormV3/ReadVariableOp_1�
!CatDogModel/bn_1/FusedBatchNormV3FusedBatchNormV3%CatDogModel/conv2D_1/BiasAdd:output:0'CatDogModel/bn_1/ReadVariableOp:value:0)CatDogModel/bn_1/ReadVariableOp_1:value:08CatDogModel/bn_1/FusedBatchNormV3/ReadVariableOp:value:0:CatDogModel/bn_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 2#
!CatDogModel/bn_1/FusedBatchNormV3�
CatDogModel/activation_1/ReluRelu%CatDogModel/bn_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������@2
CatDogModel/activation_1/Relu�
CatDogModel/max_pool_1/MaxPoolMaxPool+CatDogModel/activation_1/Relu:activations:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
2 
CatDogModel/max_pool_1/MaxPool�
CatDogModel/dropout_1/IdentityIdentity'CatDogModel/max_pool_1/MaxPool:output:0*
T0*/
_output_shapes
:���������@2 
CatDogModel/dropout_1/Identity�
*CatDogModel/conv2D_2/Conv2D/ReadVariableOpReadVariableOp3catdogmodel_conv2d_2_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype02,
*CatDogModel/conv2D_2/Conv2D/ReadVariableOp�
CatDogModel/conv2D_2/Conv2DConv2D'CatDogModel/dropout_1/Identity:output:02CatDogModel/conv2D_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
2
CatDogModel/conv2D_2/Conv2D�
+CatDogModel/conv2D_2/BiasAdd/ReadVariableOpReadVariableOp4catdogmodel_conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+CatDogModel/conv2D_2/BiasAdd/ReadVariableOp�
CatDogModel/conv2D_2/BiasAddBiasAdd$CatDogModel/conv2D_2/Conv2D:output:03CatDogModel/conv2D_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
CatDogModel/conv2D_2/BiasAdd�
CatDogModel/bn_2/ReadVariableOpReadVariableOp(catdogmodel_bn_2_readvariableop_resource*
_output_shapes	
:�*
dtype02!
CatDogModel/bn_2/ReadVariableOp�
!CatDogModel/bn_2/ReadVariableOp_1ReadVariableOp*catdogmodel_bn_2_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!CatDogModel/bn_2/ReadVariableOp_1�
0CatDogModel/bn_2/FusedBatchNormV3/ReadVariableOpReadVariableOp9catdogmodel_bn_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype022
0CatDogModel/bn_2/FusedBatchNormV3/ReadVariableOp�
2CatDogModel/bn_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp;catdogmodel_bn_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype024
2CatDogModel/bn_2/FusedBatchNormV3/ReadVariableOp_1�
!CatDogModel/bn_2/FusedBatchNormV3FusedBatchNormV3%CatDogModel/conv2D_2/BiasAdd:output:0'CatDogModel/bn_2/ReadVariableOp:value:0)CatDogModel/bn_2/ReadVariableOp_1:value:08CatDogModel/bn_2/FusedBatchNormV3/ReadVariableOp:value:0:CatDogModel/bn_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( 2#
!CatDogModel/bn_2/FusedBatchNormV3�
CatDogModel/activation_2/ReluRelu%CatDogModel/bn_2/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:����������2
CatDogModel/activation_2/Relu�
CatDogModel/max_pool_2/MaxPoolMaxPool+CatDogModel/activation_2/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2 
CatDogModel/max_pool_2/MaxPool�
CatDogModel/dropout_2/IdentityIdentity'CatDogModel/max_pool_2/MaxPool:output:0*
T0*0
_output_shapes
:����������2 
CatDogModel/dropout_2/Identity�
CatDogModel/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2
CatDogModel/flatten/Const�
CatDogModel/flatten/ReshapeReshape'CatDogModel/dropout_2/Identity:output:0"CatDogModel/flatten/Const:output:0*
T0*(
_output_shapes
:����������$2
CatDogModel/flatten/Reshape�
&CatDogModel/fc_0/MatMul/ReadVariableOpReadVariableOp/catdogmodel_fc_0_matmul_readvariableop_resource* 
_output_shapes
:
�$�*
dtype02(
&CatDogModel/fc_0/MatMul/ReadVariableOp�
CatDogModel/fc_0/MatMulMatMul$CatDogModel/flatten/Reshape:output:0.CatDogModel/fc_0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
CatDogModel/fc_0/MatMul�
'CatDogModel/fc_0/BiasAdd/ReadVariableOpReadVariableOp0catdogmodel_fc_0_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02)
'CatDogModel/fc_0/BiasAdd/ReadVariableOp�
CatDogModel/fc_0/BiasAddBiasAdd!CatDogModel/fc_0/MatMul:product:0/CatDogModel/fc_0/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
CatDogModel/fc_0/BiasAdd�
CatDogModel/fc_0/ReluRelu!CatDogModel/fc_0/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
CatDogModel/fc_0/Relu�
CatDogModel/dropout_3/IdentityIdentity#CatDogModel/fc_0/Relu:activations:0*
T0*(
_output_shapes
:����������2 
CatDogModel/dropout_3/Identity�
&CatDogModel/fc_1/MatMul/ReadVariableOpReadVariableOp/catdogmodel_fc_1_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02(
&CatDogModel/fc_1/MatMul/ReadVariableOp�
CatDogModel/fc_1/MatMulMatMul'CatDogModel/dropout_3/Identity:output:0.CatDogModel/fc_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
CatDogModel/fc_1/MatMul�
'CatDogModel/fc_1/BiasAdd/ReadVariableOpReadVariableOp0catdogmodel_fc_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'CatDogModel/fc_1/BiasAdd/ReadVariableOp�
CatDogModel/fc_1/BiasAddBiasAdd!CatDogModel/fc_1/MatMul:product:0/CatDogModel/fc_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
CatDogModel/fc_1/BiasAdd�
CatDogModel/fc_1/SoftmaxSoftmax!CatDogModel/fc_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
CatDogModel/fc_1/Softmaxv
IdentityIdentity"CatDogModel/fc_1/Softmax:softmax:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapesu
s:���������@@:::::::::::::::::::::::X T
/
_output_shapes
:���������@@
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
+__inference_CatDogModel_layer_call_fn_36207

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*'
_output_shapes
:���������*8
_read_only_resource_inputs
	
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_CatDogModel_layer_call_and_return_conditional_losses_357202
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapesu
s:���������@@::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������@@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
b
D__inference_dropout_2_layer_call_and_return_conditional_losses_35355

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:����������2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:����������2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
?__inference_bn_0_layer_call_and_return_conditional_losses_34624

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� :::::i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
?__inference_bn_1_layer_call_and_return_conditional_losses_36467

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@:::::i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
$__inference_bn_0_layer_call_fn_36369

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+��������������������������� *&
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*H
fCRA
?__inference_bn_0_layer_call_and_return_conditional_losses_346242
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�$
�
?__inference_bn_0_layer_call_and_return_conditional_losses_34593

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��#AssignMovingAvg/AssignSubVariableOp�%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
?__inference_bn_2_layer_call_and_return_conditional_losses_34942

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������:::::j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�	
�
C__inference_conv2D_0_layer_call_and_return_conditional_losses_34499

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+��������������������������� *
paddingVALID*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+��������������������������� 2	
BiasAdd~
IdentityIdentityBiasAdd:output:0*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������:::i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
^
B__inference_flatten_layer_call_and_return_conditional_losses_36810

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������$2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������$2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
y
$__inference_fc_1_layer_call_fn_36882

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:���������*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*H
fCRA
?__inference_fc_1_layer_call_and_return_conditional_losses_354502
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
`
B__inference_dropout_layer_call_and_return_conditional_losses_36396

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:��������� 2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:��������� 2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:��������� :W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�$
�
?__inference_bn_1_layer_call_and_return_conditional_losses_35137

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��#AssignMovingAvg/AssignSubVariableOp�%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
F
*__inference_max_pool_1_layer_call_fn_34806

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_max_pool_1_layer_call_and_return_conditional_losses_348002
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
$__inference_bn_1_layer_call_fn_36493

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+���������������������������@*&
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*H
fCRA
?__inference_bn_1_layer_call_and_return_conditional_losses_347832
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
F
*__inference_max_pool_2_layer_call_fn_34965

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_max_pool_2_layer_call_and_return_conditional_losses_349592
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�K
�
F__inference_CatDogModel_layer_call_and_return_conditional_losses_35720

inputs
conv2d_0_35656
conv2d_0_35658

bn_0_35661

bn_0_35663

bn_0_35665

bn_0_35667
conv2d_1_35673
conv2d_1_35675

bn_1_35678

bn_1_35680

bn_1_35682

bn_1_35684
conv2d_2_35690
conv2d_2_35692

bn_2_35695

bn_2_35697

bn_2_35699

bn_2_35701

fc_0_35708

fc_0_35710

fc_1_35714

fc_1_35716
identity��bn_0/StatefulPartitionedCall�bn_1/StatefulPartitionedCall�bn_2/StatefulPartitionedCall� conv2D_0/StatefulPartitionedCall� conv2D_1/StatefulPartitionedCall� conv2D_2/StatefulPartitionedCall�fc_0/StatefulPartitionedCall�fc_1/StatefulPartitionedCall�
 conv2D_0/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_0_35656conv2d_0_35658*
Tin
2*
Tout
2*/
_output_shapes
:���������>> *$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_conv2D_0_layer_call_and_return_conditional_losses_344992"
 conv2D_0/StatefulPartitionedCall�
bn_0/StatefulPartitionedCallStatefulPartitionedCall)conv2D_0/StatefulPartitionedCall:output:0
bn_0_35661
bn_0_35663
bn_0_35665
bn_0_35667*
Tin	
2*
Tout
2*/
_output_shapes
:���������>> *&
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*H
fCRA
?__inference_bn_0_layer_call_and_return_conditional_losses_350222
bn_0/StatefulPartitionedCall�
activation/PartitionedCallPartitionedCall%bn_0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:���������>> * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_350632
activation/PartitionedCall�
max_pool_0/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:��������� * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_max_pool_0_layer_call_and_return_conditional_losses_346412
max_pool_0/PartitionedCall�
dropout/PartitionedCallPartitionedCall#max_pool_0/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:��������� * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_350892
dropout/PartitionedCall�
 conv2D_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0conv2d_1_35673conv2d_1_35675*
Tin
2*
Tout
2*/
_output_shapes
:���������@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_conv2D_1_layer_call_and_return_conditional_losses_346582"
 conv2D_1/StatefulPartitionedCall�
bn_1/StatefulPartitionedCallStatefulPartitionedCall)conv2D_1/StatefulPartitionedCall:output:0
bn_1_35678
bn_1_35680
bn_1_35682
bn_1_35684*
Tin	
2*
Tout
2*/
_output_shapes
:���������@*&
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*H
fCRA
?__inference_bn_1_layer_call_and_return_conditional_losses_351552
bn_1/StatefulPartitionedCall�
activation_1/PartitionedCallPartitionedCall%bn_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:���������@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_351962
activation_1/PartitionedCall�
max_pool_1/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:���������@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_max_pool_1_layer_call_and_return_conditional_losses_348002
max_pool_1/PartitionedCall�
dropout_1/PartitionedCallPartitionedCall#max_pool_1/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:���������@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_352222
dropout_1/PartitionedCall�
 conv2D_2/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0conv2d_2_35690conv2d_2_35692*
Tin
2*
Tout
2*0
_output_shapes
:����������*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_conv2D_2_layer_call_and_return_conditional_losses_348172"
 conv2D_2/StatefulPartitionedCall�
bn_2/StatefulPartitionedCallStatefulPartitionedCall)conv2D_2/StatefulPartitionedCall:output:0
bn_2_35695
bn_2_35697
bn_2_35699
bn_2_35701*
Tin	
2*
Tout
2*0
_output_shapes
:����������*&
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*H
fCRA
?__inference_bn_2_layer_call_and_return_conditional_losses_352882
bn_2/StatefulPartitionedCall�
activation_2/PartitionedCallPartitionedCall%bn_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*0
_output_shapes
:����������* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*P
fKRI
G__inference_activation_2_layer_call_and_return_conditional_losses_353292
activation_2/PartitionedCall�
max_pool_2/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*0
_output_shapes
:����������* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_max_pool_2_layer_call_and_return_conditional_losses_349592
max_pool_2/PartitionedCall�
dropout_2/PartitionedCallPartitionedCall#max_pool_2/PartitionedCall:output:0*
Tin
2*
Tout
2*0
_output_shapes
:����������* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_353552
dropout_2/PartitionedCall�
flatten/PartitionedCallPartitionedCall"dropout_2/PartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:����������$* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_353742
flatten/PartitionedCall�
fc_0/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0
fc_0_35708
fc_0_35710*
Tin
2*
Tout
2*(
_output_shapes
:����������*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*H
fCRA
?__inference_fc_0_layer_call_and_return_conditional_losses_353932
fc_0/StatefulPartitionedCall�
dropout_3/PartitionedCallPartitionedCall%fc_0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:����������* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_354262
dropout_3/PartitionedCall�
fc_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0
fc_1_35714
fc_1_35716*
Tin
2*
Tout
2*'
_output_shapes
:���������*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*H
fCRA
?__inference_fc_1_layer_call_and_return_conditional_losses_354502
fc_1/StatefulPartitionedCall�
IdentityIdentity%fc_1/StatefulPartitionedCall:output:0^bn_0/StatefulPartitionedCall^bn_1/StatefulPartitionedCall^bn_2/StatefulPartitionedCall!^conv2D_0/StatefulPartitionedCall!^conv2D_1/StatefulPartitionedCall!^conv2D_2/StatefulPartitionedCall^fc_0/StatefulPartitionedCall^fc_1/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapesu
s:���������@@::::::::::::::::::::::2<
bn_0/StatefulPartitionedCallbn_0/StatefulPartitionedCall2<
bn_1/StatefulPartitionedCallbn_1/StatefulPartitionedCall2<
bn_2/StatefulPartitionedCallbn_2/StatefulPartitionedCall2D
 conv2D_0/StatefulPartitionedCall conv2D_0/StatefulPartitionedCall2D
 conv2D_1/StatefulPartitionedCall conv2D_1/StatefulPartitionedCall2D
 conv2D_2/StatefulPartitionedCall conv2D_2/StatefulPartitionedCall2<
fc_0/StatefulPartitionedCallfc_0/StatefulPartitionedCall2<
fc_1/StatefulPartitionedCallfc_1/StatefulPartitionedCall:W S
/
_output_shapes
:���������@@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
?__inference_bn_0_layer_call_and_return_conditional_losses_35022

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������>> : : : : :*
epsilon%o�:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������>> 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������>> :::::W S
/
_output_shapes
:���������>> 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
b
D__inference_dropout_3_layer_call_and_return_conditional_losses_35426

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:����������2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
$__inference_bn_1_layer_call_fn_36568

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:���������@*&
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*H
fCRA
?__inference_bn_1_layer_call_and_return_conditional_losses_351552
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�$
�
?__inference_bn_2_layer_call_and_return_conditional_losses_35270

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��#AssignMovingAvg/AssignSubVariableOp�%AssignMovingAvg_1/AssignSubVariableOpu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:�2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:�2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:�2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:�2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:����������::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
$__inference_bn_2_layer_call_fn_36679

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*H
fCRA
?__inference_bn_2_layer_call_and_return_conditional_losses_349112
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
#__inference_signature_wrapper_35862
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*'
_output_shapes
:���������*8
_read_only_resource_inputs
	
**
config_proto

CPU

GPU 2J 8*)
f$R"
 __inference__wrapped_model_344882
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapesu
s:���������@@::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������@@
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
c
G__inference_activation_1_layer_call_and_return_conditional_losses_35196

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:���������@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
a
E__inference_max_pool_2_layer_call_and_return_conditional_losses_34959

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
2	
MaxPool�
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
?__inference_bn_1_layer_call_and_return_conditional_losses_36542

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@:::::W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
c
D__inference_dropout_3_layer_call_and_return_conditional_losses_36847

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:����������2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
$__inference_bn_1_layer_call_fn_36480

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+���������������������������@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*H
fCRA
?__inference_bn_1_layer_call_and_return_conditional_losses_347522
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
c
G__inference_activation_1_layer_call_and_return_conditional_losses_36573

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:���������@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
F
*__inference_activation_layer_call_fn_36379

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*/
_output_shapes
:���������>> * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_350632
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������>> 2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������>> :W S
/
_output_shapes
:���������>> 
 
_user_specified_nameinputs
�
`
B__inference_dropout_layer_call_and_return_conditional_losses_35089

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:��������� 2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:��������� 2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:��������� :W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
^
B__inference_flatten_layer_call_and_return_conditional_losses_35374

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������$2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������$2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�$
�
?__inference_bn_1_layer_call_and_return_conditional_losses_36449

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��#AssignMovingAvg/AssignSubVariableOp�%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
F
*__inference_max_pool_0_layer_call_fn_34647

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_max_pool_0_layer_call_and_return_conditional_losses_346412
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
H
,__inference_activation_2_layer_call_fn_36777

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*0
_output_shapes
:����������* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*P
fKRI
G__inference_activation_2_layer_call_and_return_conditional_losses_353292
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�$
�
?__inference_bn_0_layer_call_and_return_conditional_losses_36250

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��#AssignMovingAvg/AssignSubVariableOp�%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������>> : : : : :*
epsilon%o�:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*/
_output_shapes
:���������>> 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������>> ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:���������>> 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�$
�
?__inference_bn_2_layer_call_and_return_conditional_losses_36648

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��#AssignMovingAvg/AssignSubVariableOp�%AssignMovingAvg_1/AssignSubVariableOpu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:�2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:�2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:�2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:�2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
+__inference_CatDogModel_layer_call_fn_35767
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*'
_output_shapes
:���������*8
_read_only_resource_inputs
	
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_CatDogModel_layer_call_and_return_conditional_losses_357202
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapesu
s:���������@@::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������@@
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_35222

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:���������@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:���������@2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
$__inference_bn_2_layer_call_fn_36767

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*0
_output_shapes
:����������*&
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*H
fCRA
?__inference_bn_2_layer_call_and_return_conditional_losses_352882
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:����������::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
E
)__inference_dropout_3_layer_call_fn_36862

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:����������* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_354262
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
a
E__inference_activation_layer_call_and_return_conditional_losses_36374

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:���������>> 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������>> 2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������>> :W S
/
_output_shapes
:���������>> 
 
_user_specified_nameinputs
�
E
)__inference_dropout_2_layer_call_fn_36804

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*0
_output_shapes
:����������* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_353552
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�S
�
F__inference_CatDogModel_layer_call_and_return_conditional_losses_36109

inputs+
'conv2d_0_conv2d_readvariableop_resource,
(conv2d_0_biasadd_readvariableop_resource 
bn_0_readvariableop_resource"
bn_0_readvariableop_1_resource1
-bn_0_fusedbatchnormv3_readvariableop_resource3
/bn_0_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource 
bn_1_readvariableop_resource"
bn_1_readvariableop_1_resource1
-bn_1_fusedbatchnormv3_readvariableop_resource3
/bn_1_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource 
bn_2_readvariableop_resource"
bn_2_readvariableop_1_resource1
-bn_2_fusedbatchnormv3_readvariableop_resource3
/bn_2_fusedbatchnormv3_readvariableop_1_resource'
#fc_0_matmul_readvariableop_resource(
$fc_0_biasadd_readvariableop_resource'
#fc_1_matmul_readvariableop_resource(
$fc_1_biasadd_readvariableop_resource
identity��
conv2D_0/Conv2D/ReadVariableOpReadVariableOp'conv2d_0_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2D_0/Conv2D/ReadVariableOp�
conv2D_0/Conv2DConv2Dinputs&conv2D_0/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������>> *
paddingVALID*
strides
2
conv2D_0/Conv2D�
conv2D_0/BiasAdd/ReadVariableOpReadVariableOp(conv2d_0_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2D_0/BiasAdd/ReadVariableOp�
conv2D_0/BiasAddBiasAddconv2D_0/Conv2D:output:0'conv2D_0/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������>> 2
conv2D_0/BiasAdd�
bn_0/ReadVariableOpReadVariableOpbn_0_readvariableop_resource*
_output_shapes
: *
dtype02
bn_0/ReadVariableOp�
bn_0/ReadVariableOp_1ReadVariableOpbn_0_readvariableop_1_resource*
_output_shapes
: *
dtype02
bn_0/ReadVariableOp_1�
$bn_0/FusedBatchNormV3/ReadVariableOpReadVariableOp-bn_0_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02&
$bn_0/FusedBatchNormV3/ReadVariableOp�
&bn_0/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp/bn_0_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&bn_0/FusedBatchNormV3/ReadVariableOp_1�
bn_0/FusedBatchNormV3FusedBatchNormV3conv2D_0/BiasAdd:output:0bn_0/ReadVariableOp:value:0bn_0/ReadVariableOp_1:value:0,bn_0/FusedBatchNormV3/ReadVariableOp:value:0.bn_0/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������>> : : : : :*
epsilon%o�:*
is_training( 2
bn_0/FusedBatchNormV3
activation/ReluRelubn_0/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������>> 2
activation/Relu�
max_pool_0/MaxPoolMaxPoolactivation/Relu:activations:0*/
_output_shapes
:��������� *
ksize
*
paddingVALID*
strides
2
max_pool_0/MaxPool�
dropout/IdentityIdentitymax_pool_0/MaxPool:output:0*
T0*/
_output_shapes
:��������� 2
dropout/Identity�
conv2D_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
conv2D_1/Conv2D/ReadVariableOp�
conv2D_1/Conv2DConv2Ddropout/Identity:output:0&conv2D_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingVALID*
strides
2
conv2D_1/Conv2D�
conv2D_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2D_1/BiasAdd/ReadVariableOp�
conv2D_1/BiasAddBiasAddconv2D_1/Conv2D:output:0'conv2D_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
conv2D_1/BiasAdd�
bn_1/ReadVariableOpReadVariableOpbn_1_readvariableop_resource*
_output_shapes
:@*
dtype02
bn_1/ReadVariableOp�
bn_1/ReadVariableOp_1ReadVariableOpbn_1_readvariableop_1_resource*
_output_shapes
:@*
dtype02
bn_1/ReadVariableOp_1�
$bn_1/FusedBatchNormV3/ReadVariableOpReadVariableOp-bn_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02&
$bn_1/FusedBatchNormV3/ReadVariableOp�
&bn_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp/bn_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&bn_1/FusedBatchNormV3/ReadVariableOp_1�
bn_1/FusedBatchNormV3FusedBatchNormV3conv2D_1/BiasAdd:output:0bn_1/ReadVariableOp:value:0bn_1/ReadVariableOp_1:value:0,bn_1/FusedBatchNormV3/ReadVariableOp:value:0.bn_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
bn_1/FusedBatchNormV3�
activation_1/ReluRelubn_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������@2
activation_1/Relu�
max_pool_1/MaxPoolMaxPoolactivation_1/Relu:activations:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
2
max_pool_1/MaxPool�
dropout_1/IdentityIdentitymax_pool_1/MaxPool:output:0*
T0*/
_output_shapes
:���������@2
dropout_1/Identity�
conv2D_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype02 
conv2D_2/Conv2D/ReadVariableOp�
conv2D_2/Conv2DConv2Ddropout_1/Identity:output:0&conv2D_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
2
conv2D_2/Conv2D�
conv2D_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
conv2D_2/BiasAdd/ReadVariableOp�
conv2D_2/BiasAddBiasAddconv2D_2/Conv2D:output:0'conv2D_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2D_2/BiasAdd�
bn_2/ReadVariableOpReadVariableOpbn_2_readvariableop_resource*
_output_shapes	
:�*
dtype02
bn_2/ReadVariableOp�
bn_2/ReadVariableOp_1ReadVariableOpbn_2_readvariableop_1_resource*
_output_shapes	
:�*
dtype02
bn_2/ReadVariableOp_1�
$bn_2/FusedBatchNormV3/ReadVariableOpReadVariableOp-bn_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02&
$bn_2/FusedBatchNormV3/ReadVariableOp�
&bn_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp/bn_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02(
&bn_2/FusedBatchNormV3/ReadVariableOp_1�
bn_2/FusedBatchNormV3FusedBatchNormV3conv2D_2/BiasAdd:output:0bn_2/ReadVariableOp:value:0bn_2/ReadVariableOp_1:value:0,bn_2/FusedBatchNormV3/ReadVariableOp:value:0.bn_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( 2
bn_2/FusedBatchNormV3�
activation_2/ReluRelubn_2/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:����������2
activation_2/Relu�
max_pool_2/MaxPoolMaxPoolactivation_2/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2
max_pool_2/MaxPool�
dropout_2/IdentityIdentitymax_pool_2/MaxPool:output:0*
T0*0
_output_shapes
:����������2
dropout_2/Identityo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2
flatten/Const�
flatten/ReshapeReshapedropout_2/Identity:output:0flatten/Const:output:0*
T0*(
_output_shapes
:����������$2
flatten/Reshape�
fc_0/MatMul/ReadVariableOpReadVariableOp#fc_0_matmul_readvariableop_resource* 
_output_shapes
:
�$�*
dtype02
fc_0/MatMul/ReadVariableOp�
fc_0/MatMulMatMulflatten/Reshape:output:0"fc_0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
fc_0/MatMul�
fc_0/BiasAdd/ReadVariableOpReadVariableOp$fc_0_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
fc_0/BiasAdd/ReadVariableOp�
fc_0/BiasAddBiasAddfc_0/MatMul:product:0#fc_0/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
fc_0/BiasAddh
	fc_0/ReluRelufc_0/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
	fc_0/Relu�
dropout_3/IdentityIdentityfc_0/Relu:activations:0*
T0*(
_output_shapes
:����������2
dropout_3/Identity�
fc_1/MatMul/ReadVariableOpReadVariableOp#fc_1_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
fc_1/MatMul/ReadVariableOp�
fc_1/MatMulMatMuldropout_3/Identity:output:0"fc_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
fc_1/MatMul�
fc_1/BiasAdd/ReadVariableOpReadVariableOp$fc_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
fc_1/BiasAdd/ReadVariableOp�
fc_1/BiasAddBiasAddfc_1/MatMul:product:0#fc_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
fc_1/BiasAddp
fc_1/SoftmaxSoftmaxfc_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
fc_1/Softmaxj
IdentityIdentityfc_1/Softmax:softmax:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapesu
s:���������@@:::::::::::::::::::::::W S
/
_output_shapes
:���������@@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
?__inference_fc_0_layer_call_and_return_conditional_losses_35393

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
�$�*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������$:::P L
(
_output_shapes
:����������$
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
�
?__inference_bn_2_layer_call_and_return_conditional_losses_35288

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:����������:::::X T
0
_output_shapes
:����������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
C
'__inference_flatten_layer_call_fn_36815

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:����������$* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_353742
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������$2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
$__inference_bn_2_layer_call_fn_36754

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*0
_output_shapes
:����������*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*H
fCRA
?__inference_bn_2_layer_call_and_return_conditional_losses_352702
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:����������::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
a
E__inference_activation_layer_call_and_return_conditional_losses_35063

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:���������>> 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������>> 2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������>> :W S
/
_output_shapes
:���������>> 
 
_user_specified_nameinputs
�$
�
?__inference_bn_0_layer_call_and_return_conditional_losses_36325

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��#AssignMovingAvg/AssignSubVariableOp�%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
?__inference_bn_1_layer_call_and_return_conditional_losses_34783

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@:::::i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
}
(__inference_conv2D_1_layer_call_fn_34668

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+���������������������������@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_conv2D_1_layer_call_and_return_conditional_losses_346582
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+��������������������������� ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
��
�

F__inference_CatDogModel_layer_call_and_return_conditional_losses_36019

inputs+
'conv2d_0_conv2d_readvariableop_resource,
(conv2d_0_biasadd_readvariableop_resource 
bn_0_readvariableop_resource"
bn_0_readvariableop_1_resource1
-bn_0_fusedbatchnormv3_readvariableop_resource3
/bn_0_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource 
bn_1_readvariableop_resource"
bn_1_readvariableop_1_resource1
-bn_1_fusedbatchnormv3_readvariableop_resource3
/bn_1_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource 
bn_2_readvariableop_resource"
bn_2_readvariableop_1_resource1
-bn_2_fusedbatchnormv3_readvariableop_resource3
/bn_2_fusedbatchnormv3_readvariableop_1_resource'
#fc_0_matmul_readvariableop_resource(
$fc_0_biasadd_readvariableop_resource'
#fc_1_matmul_readvariableop_resource(
$fc_1_biasadd_readvariableop_resource
identity��(bn_0/AssignMovingAvg/AssignSubVariableOp�*bn_0/AssignMovingAvg_1/AssignSubVariableOp�(bn_1/AssignMovingAvg/AssignSubVariableOp�*bn_1/AssignMovingAvg_1/AssignSubVariableOp�(bn_2/AssignMovingAvg/AssignSubVariableOp�*bn_2/AssignMovingAvg_1/AssignSubVariableOp�
conv2D_0/Conv2D/ReadVariableOpReadVariableOp'conv2d_0_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2D_0/Conv2D/ReadVariableOp�
conv2D_0/Conv2DConv2Dinputs&conv2D_0/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������>> *
paddingVALID*
strides
2
conv2D_0/Conv2D�
conv2D_0/BiasAdd/ReadVariableOpReadVariableOp(conv2d_0_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2D_0/BiasAdd/ReadVariableOp�
conv2D_0/BiasAddBiasAddconv2D_0/Conv2D:output:0'conv2D_0/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������>> 2
conv2D_0/BiasAdd�
bn_0/ReadVariableOpReadVariableOpbn_0_readvariableop_resource*
_output_shapes
: *
dtype02
bn_0/ReadVariableOp�
bn_0/ReadVariableOp_1ReadVariableOpbn_0_readvariableop_1_resource*
_output_shapes
: *
dtype02
bn_0/ReadVariableOp_1�
$bn_0/FusedBatchNormV3/ReadVariableOpReadVariableOp-bn_0_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02&
$bn_0/FusedBatchNormV3/ReadVariableOp�
&bn_0/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp/bn_0_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&bn_0/FusedBatchNormV3/ReadVariableOp_1�
bn_0/FusedBatchNormV3FusedBatchNormV3conv2D_0/BiasAdd:output:0bn_0/ReadVariableOp:value:0bn_0/ReadVariableOp_1:value:0,bn_0/FusedBatchNormV3/ReadVariableOp:value:0.bn_0/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������>> : : : : :*
epsilon%o�:2
bn_0/FusedBatchNormV3]

bn_0/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2

bn_0/Const�
bn_0/AssignMovingAvg/sub/xConst*@
_class6
42loc:@bn_0/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2
bn_0/AssignMovingAvg/sub/x�
bn_0/AssignMovingAvg/subSub#bn_0/AssignMovingAvg/sub/x:output:0bn_0/Const:output:0*
T0*@
_class6
42loc:@bn_0/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
bn_0/AssignMovingAvg/sub�
#bn_0/AssignMovingAvg/ReadVariableOpReadVariableOp-bn_0_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02%
#bn_0/AssignMovingAvg/ReadVariableOp�
bn_0/AssignMovingAvg/sub_1Sub+bn_0/AssignMovingAvg/ReadVariableOp:value:0"bn_0/FusedBatchNormV3:batch_mean:0*
T0*@
_class6
42loc:@bn_0/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
bn_0/AssignMovingAvg/sub_1�
bn_0/AssignMovingAvg/mulMulbn_0/AssignMovingAvg/sub_1:z:0bn_0/AssignMovingAvg/sub:z:0*
T0*@
_class6
42loc:@bn_0/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
bn_0/AssignMovingAvg/mul�
(bn_0/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp-bn_0_fusedbatchnormv3_readvariableop_resourcebn_0/AssignMovingAvg/mul:z:0$^bn_0/AssignMovingAvg/ReadVariableOp%^bn_0/FusedBatchNormV3/ReadVariableOp*@
_class6
42loc:@bn_0/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02*
(bn_0/AssignMovingAvg/AssignSubVariableOp�
bn_0/AssignMovingAvg_1/sub/xConst*B
_class8
64loc:@bn_0/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2
bn_0/AssignMovingAvg_1/sub/x�
bn_0/AssignMovingAvg_1/subSub%bn_0/AssignMovingAvg_1/sub/x:output:0bn_0/Const:output:0*
T0*B
_class8
64loc:@bn_0/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
bn_0/AssignMovingAvg_1/sub�
%bn_0/AssignMovingAvg_1/ReadVariableOpReadVariableOp/bn_0_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02'
%bn_0/AssignMovingAvg_1/ReadVariableOp�
bn_0/AssignMovingAvg_1/sub_1Sub-bn_0/AssignMovingAvg_1/ReadVariableOp:value:0&bn_0/FusedBatchNormV3:batch_variance:0*
T0*B
_class8
64loc:@bn_0/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
bn_0/AssignMovingAvg_1/sub_1�
bn_0/AssignMovingAvg_1/mulMul bn_0/AssignMovingAvg_1/sub_1:z:0bn_0/AssignMovingAvg_1/sub:z:0*
T0*B
_class8
64loc:@bn_0/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
bn_0/AssignMovingAvg_1/mul�
*bn_0/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp/bn_0_fusedbatchnormv3_readvariableop_1_resourcebn_0/AssignMovingAvg_1/mul:z:0&^bn_0/AssignMovingAvg_1/ReadVariableOp'^bn_0/FusedBatchNormV3/ReadVariableOp_1*B
_class8
64loc:@bn_0/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02,
*bn_0/AssignMovingAvg_1/AssignSubVariableOp
activation/ReluRelubn_0/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������>> 2
activation/Relu�
max_pool_0/MaxPoolMaxPoolactivation/Relu:activations:0*/
_output_shapes
:��������� *
ksize
*
paddingVALID*
strides
2
max_pool_0/MaxPools
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/dropout/Const�
dropout/dropout/MulMulmax_pool_0/MaxPool:output:0dropout/dropout/Const:output:0*
T0*/
_output_shapes
:��������� 2
dropout/dropout/Muly
dropout/dropout/ShapeShapemax_pool_0/MaxPool:output:0*
T0*
_output_shapes
:2
dropout/dropout/Shape�
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*/
_output_shapes
:��������� *
dtype02.
,dropout/dropout/random_uniform/RandomUniform�
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2 
dropout/dropout/GreaterEqual/y�
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:��������� 2
dropout/dropout/GreaterEqual�
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:��������� 2
dropout/dropout/Cast�
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:��������� 2
dropout/dropout/Mul_1�
conv2D_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
conv2D_1/Conv2D/ReadVariableOp�
conv2D_1/Conv2DConv2Ddropout/dropout/Mul_1:z:0&conv2D_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingVALID*
strides
2
conv2D_1/Conv2D�
conv2D_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2D_1/BiasAdd/ReadVariableOp�
conv2D_1/BiasAddBiasAddconv2D_1/Conv2D:output:0'conv2D_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
conv2D_1/BiasAdd�
bn_1/ReadVariableOpReadVariableOpbn_1_readvariableop_resource*
_output_shapes
:@*
dtype02
bn_1/ReadVariableOp�
bn_1/ReadVariableOp_1ReadVariableOpbn_1_readvariableop_1_resource*
_output_shapes
:@*
dtype02
bn_1/ReadVariableOp_1�
$bn_1/FusedBatchNormV3/ReadVariableOpReadVariableOp-bn_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02&
$bn_1/FusedBatchNormV3/ReadVariableOp�
&bn_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp/bn_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&bn_1/FusedBatchNormV3/ReadVariableOp_1�
bn_1/FusedBatchNormV3FusedBatchNormV3conv2D_1/BiasAdd:output:0bn_1/ReadVariableOp:value:0bn_1/ReadVariableOp_1:value:0,bn_1/FusedBatchNormV3/ReadVariableOp:value:0.bn_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:2
bn_1/FusedBatchNormV3]

bn_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2

bn_1/Const�
bn_1/AssignMovingAvg/sub/xConst*@
_class6
42loc:@bn_1/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2
bn_1/AssignMovingAvg/sub/x�
bn_1/AssignMovingAvg/subSub#bn_1/AssignMovingAvg/sub/x:output:0bn_1/Const:output:0*
T0*@
_class6
42loc:@bn_1/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
bn_1/AssignMovingAvg/sub�
#bn_1/AssignMovingAvg/ReadVariableOpReadVariableOp-bn_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02%
#bn_1/AssignMovingAvg/ReadVariableOp�
bn_1/AssignMovingAvg/sub_1Sub+bn_1/AssignMovingAvg/ReadVariableOp:value:0"bn_1/FusedBatchNormV3:batch_mean:0*
T0*@
_class6
42loc:@bn_1/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
bn_1/AssignMovingAvg/sub_1�
bn_1/AssignMovingAvg/mulMulbn_1/AssignMovingAvg/sub_1:z:0bn_1/AssignMovingAvg/sub:z:0*
T0*@
_class6
42loc:@bn_1/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
bn_1/AssignMovingAvg/mul�
(bn_1/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp-bn_1_fusedbatchnormv3_readvariableop_resourcebn_1/AssignMovingAvg/mul:z:0$^bn_1/AssignMovingAvg/ReadVariableOp%^bn_1/FusedBatchNormV3/ReadVariableOp*@
_class6
42loc:@bn_1/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02*
(bn_1/AssignMovingAvg/AssignSubVariableOp�
bn_1/AssignMovingAvg_1/sub/xConst*B
_class8
64loc:@bn_1/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2
bn_1/AssignMovingAvg_1/sub/x�
bn_1/AssignMovingAvg_1/subSub%bn_1/AssignMovingAvg_1/sub/x:output:0bn_1/Const:output:0*
T0*B
_class8
64loc:@bn_1/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
bn_1/AssignMovingAvg_1/sub�
%bn_1/AssignMovingAvg_1/ReadVariableOpReadVariableOp/bn_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02'
%bn_1/AssignMovingAvg_1/ReadVariableOp�
bn_1/AssignMovingAvg_1/sub_1Sub-bn_1/AssignMovingAvg_1/ReadVariableOp:value:0&bn_1/FusedBatchNormV3:batch_variance:0*
T0*B
_class8
64loc:@bn_1/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
bn_1/AssignMovingAvg_1/sub_1�
bn_1/AssignMovingAvg_1/mulMul bn_1/AssignMovingAvg_1/sub_1:z:0bn_1/AssignMovingAvg_1/sub:z:0*
T0*B
_class8
64loc:@bn_1/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
bn_1/AssignMovingAvg_1/mul�
*bn_1/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp/bn_1_fusedbatchnormv3_readvariableop_1_resourcebn_1/AssignMovingAvg_1/mul:z:0&^bn_1/AssignMovingAvg_1/ReadVariableOp'^bn_1/FusedBatchNormV3/ReadVariableOp_1*B
_class8
64loc:@bn_1/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02,
*bn_1/AssignMovingAvg_1/AssignSubVariableOp�
activation_1/ReluRelubn_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������@2
activation_1/Relu�
max_pool_1/MaxPoolMaxPoolactivation_1/Relu:activations:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
2
max_pool_1/MaxPoolw
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout_1/dropout/Const�
dropout_1/dropout/MulMulmax_pool_1/MaxPool:output:0 dropout_1/dropout/Const:output:0*
T0*/
_output_shapes
:���������@2
dropout_1/dropout/Mul}
dropout_1/dropout/ShapeShapemax_pool_1/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_1/dropout/Shape�
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*/
_output_shapes
:���������@*
dtype020
.dropout_1/dropout/random_uniform/RandomUniform�
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2"
 dropout_1/dropout/GreaterEqual/y�
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������@2 
dropout_1/dropout/GreaterEqual�
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:���������@2
dropout_1/dropout/Cast�
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*/
_output_shapes
:���������@2
dropout_1/dropout/Mul_1�
conv2D_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype02 
conv2D_2/Conv2D/ReadVariableOp�
conv2D_2/Conv2DConv2Ddropout_1/dropout/Mul_1:z:0&conv2D_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
2
conv2D_2/Conv2D�
conv2D_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
conv2D_2/BiasAdd/ReadVariableOp�
conv2D_2/BiasAddBiasAddconv2D_2/Conv2D:output:0'conv2D_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2D_2/BiasAdd�
bn_2/ReadVariableOpReadVariableOpbn_2_readvariableop_resource*
_output_shapes	
:�*
dtype02
bn_2/ReadVariableOp�
bn_2/ReadVariableOp_1ReadVariableOpbn_2_readvariableop_1_resource*
_output_shapes	
:�*
dtype02
bn_2/ReadVariableOp_1�
$bn_2/FusedBatchNormV3/ReadVariableOpReadVariableOp-bn_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02&
$bn_2/FusedBatchNormV3/ReadVariableOp�
&bn_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp/bn_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02(
&bn_2/FusedBatchNormV3/ReadVariableOp_1�
bn_2/FusedBatchNormV3FusedBatchNormV3conv2D_2/BiasAdd:output:0bn_2/ReadVariableOp:value:0bn_2/ReadVariableOp_1:value:0,bn_2/FusedBatchNormV3/ReadVariableOp:value:0.bn_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:2
bn_2/FusedBatchNormV3]

bn_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2

bn_2/Const�
bn_2/AssignMovingAvg/sub/xConst*@
_class6
42loc:@bn_2/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2
bn_2/AssignMovingAvg/sub/x�
bn_2/AssignMovingAvg/subSub#bn_2/AssignMovingAvg/sub/x:output:0bn_2/Const:output:0*
T0*@
_class6
42loc:@bn_2/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
bn_2/AssignMovingAvg/sub�
#bn_2/AssignMovingAvg/ReadVariableOpReadVariableOp-bn_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#bn_2/AssignMovingAvg/ReadVariableOp�
bn_2/AssignMovingAvg/sub_1Sub+bn_2/AssignMovingAvg/ReadVariableOp:value:0"bn_2/FusedBatchNormV3:batch_mean:0*
T0*@
_class6
42loc:@bn_2/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:�2
bn_2/AssignMovingAvg/sub_1�
bn_2/AssignMovingAvg/mulMulbn_2/AssignMovingAvg/sub_1:z:0bn_2/AssignMovingAvg/sub:z:0*
T0*@
_class6
42loc:@bn_2/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:�2
bn_2/AssignMovingAvg/mul�
(bn_2/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp-bn_2_fusedbatchnormv3_readvariableop_resourcebn_2/AssignMovingAvg/mul:z:0$^bn_2/AssignMovingAvg/ReadVariableOp%^bn_2/FusedBatchNormV3/ReadVariableOp*@
_class6
42loc:@bn_2/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02*
(bn_2/AssignMovingAvg/AssignSubVariableOp�
bn_2/AssignMovingAvg_1/sub/xConst*B
_class8
64loc:@bn_2/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2
bn_2/AssignMovingAvg_1/sub/x�
bn_2/AssignMovingAvg_1/subSub%bn_2/AssignMovingAvg_1/sub/x:output:0bn_2/Const:output:0*
T0*B
_class8
64loc:@bn_2/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
bn_2/AssignMovingAvg_1/sub�
%bn_2/AssignMovingAvg_1/ReadVariableOpReadVariableOp/bn_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02'
%bn_2/AssignMovingAvg_1/ReadVariableOp�
bn_2/AssignMovingAvg_1/sub_1Sub-bn_2/AssignMovingAvg_1/ReadVariableOp:value:0&bn_2/FusedBatchNormV3:batch_variance:0*
T0*B
_class8
64loc:@bn_2/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:�2
bn_2/AssignMovingAvg_1/sub_1�
bn_2/AssignMovingAvg_1/mulMul bn_2/AssignMovingAvg_1/sub_1:z:0bn_2/AssignMovingAvg_1/sub:z:0*
T0*B
_class8
64loc:@bn_2/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:�2
bn_2/AssignMovingAvg_1/mul�
*bn_2/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp/bn_2_fusedbatchnormv3_readvariableop_1_resourcebn_2/AssignMovingAvg_1/mul:z:0&^bn_2/AssignMovingAvg_1/ReadVariableOp'^bn_2/FusedBatchNormV3/ReadVariableOp_1*B
_class8
64loc:@bn_2/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02,
*bn_2/AssignMovingAvg_1/AssignSubVariableOp�
activation_2/ReluRelubn_2/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:����������2
activation_2/Relu�
max_pool_2/MaxPoolMaxPoolactivation_2/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2
max_pool_2/MaxPoolw
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout_2/dropout/Const�
dropout_2/dropout/MulMulmax_pool_2/MaxPool:output:0 dropout_2/dropout/Const:output:0*
T0*0
_output_shapes
:����������2
dropout_2/dropout/Mul}
dropout_2/dropout/ShapeShapemax_pool_2/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_2/dropout/Shape�
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*0
_output_shapes
:����������*
dtype020
.dropout_2/dropout/random_uniform/RandomUniform�
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2"
 dropout_2/dropout/GreaterEqual/y�
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:����������2 
dropout_2/dropout/GreaterEqual�
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:����������2
dropout_2/dropout/Cast�
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*0
_output_shapes
:����������2
dropout_2/dropout/Mul_1o
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2
flatten/Const�
flatten/ReshapeReshapedropout_2/dropout/Mul_1:z:0flatten/Const:output:0*
T0*(
_output_shapes
:����������$2
flatten/Reshape�
fc_0/MatMul/ReadVariableOpReadVariableOp#fc_0_matmul_readvariableop_resource* 
_output_shapes
:
�$�*
dtype02
fc_0/MatMul/ReadVariableOp�
fc_0/MatMulMatMulflatten/Reshape:output:0"fc_0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
fc_0/MatMul�
fc_0/BiasAdd/ReadVariableOpReadVariableOp$fc_0_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
fc_0/BiasAdd/ReadVariableOp�
fc_0/BiasAddBiasAddfc_0/MatMul:product:0#fc_0/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
fc_0/BiasAddh
	fc_0/ReluRelufc_0/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
	fc_0/Reluw
dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_3/dropout/Const�
dropout_3/dropout/MulMulfc_0/Relu:activations:0 dropout_3/dropout/Const:output:0*
T0*(
_output_shapes
:����������2
dropout_3/dropout/Muly
dropout_3/dropout/ShapeShapefc_0/Relu:activations:0*
T0*
_output_shapes
:2
dropout_3/dropout/Shape�
.dropout_3/dropout/random_uniform/RandomUniformRandomUniform dropout_3/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype020
.dropout_3/dropout/random_uniform/RandomUniform�
 dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 dropout_3/dropout/GreaterEqual/y�
dropout_3/dropout/GreaterEqualGreaterEqual7dropout_3/dropout/random_uniform/RandomUniform:output:0)dropout_3/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2 
dropout_3/dropout/GreaterEqual�
dropout_3/dropout/CastCast"dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2
dropout_3/dropout/Cast�
dropout_3/dropout/Mul_1Muldropout_3/dropout/Mul:z:0dropout_3/dropout/Cast:y:0*
T0*(
_output_shapes
:����������2
dropout_3/dropout/Mul_1�
fc_1/MatMul/ReadVariableOpReadVariableOp#fc_1_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
fc_1/MatMul/ReadVariableOp�
fc_1/MatMulMatMuldropout_3/dropout/Mul_1:z:0"fc_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
fc_1/MatMul�
fc_1/BiasAdd/ReadVariableOpReadVariableOp$fc_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
fc_1/BiasAdd/ReadVariableOp�
fc_1/BiasAddBiasAddfc_1/MatMul:product:0#fc_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
fc_1/BiasAddp
fc_1/SoftmaxSoftmaxfc_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
fc_1/Softmax�
IdentityIdentityfc_1/Softmax:softmax:0)^bn_0/AssignMovingAvg/AssignSubVariableOp+^bn_0/AssignMovingAvg_1/AssignSubVariableOp)^bn_1/AssignMovingAvg/AssignSubVariableOp+^bn_1/AssignMovingAvg_1/AssignSubVariableOp)^bn_2/AssignMovingAvg/AssignSubVariableOp+^bn_2/AssignMovingAvg_1/AssignSubVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapesu
s:���������@@::::::::::::::::::::::2T
(bn_0/AssignMovingAvg/AssignSubVariableOp(bn_0/AssignMovingAvg/AssignSubVariableOp2X
*bn_0/AssignMovingAvg_1/AssignSubVariableOp*bn_0/AssignMovingAvg_1/AssignSubVariableOp2T
(bn_1/AssignMovingAvg/AssignSubVariableOp(bn_1/AssignMovingAvg/AssignSubVariableOp2X
*bn_1/AssignMovingAvg_1/AssignSubVariableOp*bn_1/AssignMovingAvg_1/AssignSubVariableOp2T
(bn_2/AssignMovingAvg/AssignSubVariableOp(bn_2/AssignMovingAvg/AssignSubVariableOp2X
*bn_2/AssignMovingAvg_1/AssignSubVariableOp*bn_2/AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:���������@@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
?__inference_fc_1_layer_call_and_return_conditional_losses_36873

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:::P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
}
(__inference_conv2D_0_layer_call_fn_34509

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+��������������������������� *$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_conv2D_0_layer_call_and_return_conditional_losses_344992
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
c
G__inference_activation_2_layer_call_and_return_conditional_losses_36772

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:����������2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
?__inference_bn_2_layer_call_and_return_conditional_losses_36666

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity�u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3�
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������:::::j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
c
D__inference_dropout_2_layer_call_and_return_conditional_losses_35350

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:����������2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:����������*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:����������2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:����������2
dropout/Cast�
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:����������2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�$
�
?__inference_bn_0_layer_call_and_return_conditional_losses_35004

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��#AssignMovingAvg/AssignSubVariableOp�%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������>> : : : : :*
epsilon%o�:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*/
_output_shapes
:���������>> 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������>> ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:���������>> 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
$__inference_bn_2_layer_call_fn_36692

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*B
_output_shapes0
.:,����������������������������*&
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*H
fCRA
?__inference_bn_2_layer_call_and_return_conditional_losses_349422
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,����������������������������::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
C
'__inference_dropout_layer_call_fn_36406

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*/
_output_shapes
:��������� * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_350892
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*.
_input_shapes
:��������� :W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
H
,__inference_activation_1_layer_call_fn_36578

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*/
_output_shapes
:���������@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_351962
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�	
�
C__inference_conv2D_2_layer_call_and_return_conditional_losses_34817

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingVALID*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������2	
BiasAdd
IdentityIdentityBiasAdd:output:0*
T0*B
_output_shapes0
.:,����������������������������2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������@:::i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_36595

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:���������@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:���������@2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
a
E__inference_max_pool_0_layer_call_and_return_conditional_losses_34641

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
2	
MaxPool�
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
$__inference_bn_0_layer_call_fn_36281

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:���������>> *$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*H
fCRA
?__inference_bn_0_layer_call_and_return_conditional_losses_350042
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������>> 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������>> ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������>> 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: "�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
C
input_18
serving_default_input_1:0���������@@8
fc_10
StatefulPartitionedCall:0���������tensorflow/serving/predict:ѿ
Ǉ
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer_with_weights-3
layer-7
	layer-8

layer-9
layer-10
layer_with_weights-4
layer-11
layer_with_weights-5
layer-12
layer-13
layer-14
layer-15
layer-16
layer_with_weights-6
layer-17
layer-18
layer_with_weights-7
layer-19
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
�_default_save_signature
+�&call_and_return_all_conditional_losses
�__call__"��
_tf_keras_model��{"class_name": "Model", "name": "CatDogModel", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "CatDogModel", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2D_0", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2D_0", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "bn_0", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "bn_0", "inbound_nodes": [[["conv2D_0", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation", "inbound_nodes": [[["bn_0", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pool_0", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pool_0", "inbound_nodes": [[["activation", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["max_pool_0", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2D_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2D_1", "inbound_nodes": [[["dropout", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "bn_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "bn_1", "inbound_nodes": [[["conv2D_1", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_1", "inbound_nodes": [[["bn_1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pool_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pool_1", "inbound_nodes": [[["activation_1", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout_1", "inbound_nodes": [[["max_pool_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2D_2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2D_2", "inbound_nodes": [[["dropout_1", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "bn_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "bn_2", "inbound_nodes": [[["conv2D_2", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_2", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_2", "inbound_nodes": [[["bn_2", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pool_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pool_2", "inbound_nodes": [[["activation_2", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout_2", "inbound_nodes": [[["max_pool_2", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["dropout_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "fc_0", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "fc_0", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout_3", "inbound_nodes": [[["fc_0", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "fc_1", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "fc_1", "inbound_nodes": [[["dropout_3", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["fc_1", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 3]}, "is_graph_network": true, "keras_version": "2.3.0-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "CatDogModel", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2D_0", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2D_0", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "bn_0", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "bn_0", "inbound_nodes": [[["conv2D_0", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation", "inbound_nodes": [[["bn_0", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pool_0", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pool_0", "inbound_nodes": [[["activation", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["max_pool_0", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2D_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2D_1", "inbound_nodes": [[["dropout", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "bn_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "bn_1", "inbound_nodes": [[["conv2D_1", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_1", "inbound_nodes": [[["bn_1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pool_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pool_1", "inbound_nodes": [[["activation_1", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout_1", "inbound_nodes": [[["max_pool_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2D_2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2D_2", "inbound_nodes": [[["dropout_1", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "bn_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "bn_2", "inbound_nodes": [[["conv2D_2", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_2", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_2", "inbound_nodes": [[["bn_2", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pool_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pool_2", "inbound_nodes": [[["activation_2", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout_2", "inbound_nodes": [[["max_pool_2", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["dropout_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "fc_0", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "fc_0", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout_3", "inbound_nodes": [[["fc_0", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "fc_1", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "fc_1", "inbound_nodes": [[["dropout_3", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["fc_1", 0, 0]]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": ["accuracy"], "weighted_metrics": null, "loss_weights": null, "sample_weight_mode": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 9.999999747378752e-05, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
�	

kernel
bias
regularization_losses
trainable_variables
	variables
 	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2D_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2D_0", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 3]}}
�
!axis
	"gamma
#beta
$moving_mean
%moving_variance
&regularization_losses
'trainable_variables
(	variables
)	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "bn_0", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "bn_0", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 62, 62, 32]}}
�
*regularization_losses
+trainable_variables
,	variables
-	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Activation", "name": "activation", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}}
�
.regularization_losses
/trainable_variables
0	variables
1	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "MaxPooling2D", "name": "max_pool_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "max_pool_0", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�
2regularization_losses
3trainable_variables
4	variables
5	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
�	

6kernel
7bias
8regularization_losses
9trainable_variables
:	variables
;	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2D_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2D_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 31, 31, 32]}}
�
<axis
	=gamma
>beta
?moving_mean
@moving_variance
Aregularization_losses
Btrainable_variables
C	variables
D	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "bn_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "bn_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 29, 29, 64]}}
�
Eregularization_losses
Ftrainable_variables
G	variables
H	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Activation", "name": "activation_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}}
�
Iregularization_losses
Jtrainable_variables
K	variables
L	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "MaxPooling2D", "name": "max_pool_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "max_pool_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�
Mregularization_losses
Ntrainable_variables
O	variables
P	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dropout", "name": "dropout_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
�	

Qkernel
Rbias
Sregularization_losses
Ttrainable_variables
U	variables
V	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2D_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2D_2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 14, 14, 64]}}
�
Waxis
	Xgamma
Ybeta
Zmoving_mean
[moving_variance
\regularization_losses
]trainable_variables
^	variables
_	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "bn_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "bn_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 12, 12, 128]}}
�
`regularization_losses
atrainable_variables
b	variables
c	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Activation", "name": "activation_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "activation_2", "trainable": true, "dtype": "float32", "activation": "relu"}}
�
dregularization_losses
etrainable_variables
f	variables
g	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "MaxPooling2D", "name": "max_pool_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "max_pool_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�
hregularization_losses
itrainable_variables
j	variables
k	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dropout", "name": "dropout_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
�
lregularization_losses
mtrainable_variables
n	variables
o	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
�

pkernel
qbias
rregularization_losses
strainable_variables
t	variables
u	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "fc_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "fc_0", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 4608}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4608]}}
�
vregularization_losses
wtrainable_variables
x	variables
y	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dropout", "name": "dropout_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
�

zkernel
{bias
|regularization_losses
}trainable_variables
~	variables
	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "fc_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "fc_1", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
�
	�iter
�beta_1
�beta_2

�decay
�learning_ratem�m�"m�#m�6m�7m�=m�>m�Qm�Rm�Xm�Ym�pm�qm�zm�{m�v�v�"v�#v�6v�7v�=v�>v�Qv�Rv�Xv�Yv�pv�qv�zv�{v�"
	optimizer
 "
trackable_list_wrapper
�
0
1
"2
#3
64
75
=6
>7
Q8
R9
X10
Y11
p12
q13
z14
{15"
trackable_list_wrapper
�
0
1
"2
#3
$4
%5
66
77
=8
>9
?10
@11
Q12
R13
X14
Y15
Z16
[17
p18
q19
z20
{21"
trackable_list_wrapper
�
regularization_losses
�non_trainable_variables
�metrics
trainable_variables
�layers
�layer_metrics
 �layer_regularization_losses
	variables
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
):' 2conv2D_0/kernel
: 2conv2D_0/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
�non_trainable_variables
regularization_losses
�metrics
trainable_variables
�layers
�layer_metrics
 �layer_regularization_losses
	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
: 2
bn_0/gamma
: 2	bn_0/beta
 :  (2bn_0/moving_mean
$:"  (2bn_0/moving_variance
 "
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
<
"0
#1
$2
%3"
trackable_list_wrapper
�
�non_trainable_variables
&regularization_losses
�metrics
'trainable_variables
�layers
�layer_metrics
 �layer_regularization_losses
(	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
*regularization_losses
�metrics
+trainable_variables
�layers
�layer_metrics
 �layer_regularization_losses
,	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
.regularization_losses
�metrics
/trainable_variables
�layers
�layer_metrics
 �layer_regularization_losses
0	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
2regularization_losses
�metrics
3trainable_variables
�layers
�layer_metrics
 �layer_regularization_losses
4	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
):' @2conv2D_1/kernel
:@2conv2D_1/bias
 "
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
�
�non_trainable_variables
8regularization_losses
�metrics
9trainable_variables
�layers
�layer_metrics
 �layer_regularization_losses
:	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
:@2
bn_1/gamma
:@2	bn_1/beta
 :@ (2bn_1/moving_mean
$:"@ (2bn_1/moving_variance
 "
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
<
=0
>1
?2
@3"
trackable_list_wrapper
�
�non_trainable_variables
Aregularization_losses
�metrics
Btrainable_variables
�layers
�layer_metrics
 �layer_regularization_losses
C	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
Eregularization_losses
�metrics
Ftrainable_variables
�layers
�layer_metrics
 �layer_regularization_losses
G	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
Iregularization_losses
�metrics
Jtrainable_variables
�layers
�layer_metrics
 �layer_regularization_losses
K	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
Mregularization_losses
�metrics
Ntrainable_variables
�layers
�layer_metrics
 �layer_regularization_losses
O	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
*:(@�2conv2D_2/kernel
:�2conv2D_2/bias
 "
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
�
�non_trainable_variables
Sregularization_losses
�metrics
Ttrainable_variables
�layers
�layer_metrics
 �layer_regularization_losses
U	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
:�2
bn_2/gamma
:�2	bn_2/beta
!:� (2bn_2/moving_mean
%:#� (2bn_2/moving_variance
 "
trackable_list_wrapper
.
X0
Y1"
trackable_list_wrapper
<
X0
Y1
Z2
[3"
trackable_list_wrapper
�
�non_trainable_variables
\regularization_losses
�metrics
]trainable_variables
�layers
�layer_metrics
 �layer_regularization_losses
^	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
`regularization_losses
�metrics
atrainable_variables
�layers
�layer_metrics
 �layer_regularization_losses
b	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
dregularization_losses
�metrics
etrainable_variables
�layers
�layer_metrics
 �layer_regularization_losses
f	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
hregularization_losses
�metrics
itrainable_variables
�layers
�layer_metrics
 �layer_regularization_losses
j	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
lregularization_losses
�metrics
mtrainable_variables
�layers
�layer_metrics
 �layer_regularization_losses
n	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
:
�$�2fc_0/kernel
:�2	fc_0/bias
 "
trackable_list_wrapper
.
p0
q1"
trackable_list_wrapper
.
p0
q1"
trackable_list_wrapper
�
�non_trainable_variables
rregularization_losses
�metrics
strainable_variables
�layers
�layer_metrics
 �layer_regularization_losses
t	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
vregularization_losses
�metrics
wtrainable_variables
�layers
�layer_metrics
 �layer_regularization_losses
x	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
:	�2fc_1/kernel
:2	fc_1/bias
 "
trackable_list_wrapper
.
z0
{1"
trackable_list_wrapper
.
z0
{1"
trackable_list_wrapper
�
�non_trainable_variables
|regularization_losses
�metrics
}trainable_variables
�layers
�layer_metrics
 �layer_regularization_losses
~	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
J
$0
%1
?2
@3
Z4
[5"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
Z0
[1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
�

�total

�count
�	variables
�	keras_api"�
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
�

�total

�count
�
_fn_kwargs
�	variables
�	keras_api"�
_tf_keras_metric�{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
.:, 2Adam/conv2D_0/kernel/m
 : 2Adam/conv2D_0/bias/m
: 2Adam/bn_0/gamma/m
: 2Adam/bn_0/beta/m
.:, @2Adam/conv2D_1/kernel/m
 :@2Adam/conv2D_1/bias/m
:@2Adam/bn_1/gamma/m
:@2Adam/bn_1/beta/m
/:-@�2Adam/conv2D_2/kernel/m
!:�2Adam/conv2D_2/bias/m
:�2Adam/bn_2/gamma/m
:�2Adam/bn_2/beta/m
$:"
�$�2Adam/fc_0/kernel/m
:�2Adam/fc_0/bias/m
#:!	�2Adam/fc_1/kernel/m
:2Adam/fc_1/bias/m
.:, 2Adam/conv2D_0/kernel/v
 : 2Adam/conv2D_0/bias/v
: 2Adam/bn_0/gamma/v
: 2Adam/bn_0/beta/v
.:, @2Adam/conv2D_1/kernel/v
 :@2Adam/conv2D_1/bias/v
:@2Adam/bn_1/gamma/v
:@2Adam/bn_1/beta/v
/:-@�2Adam/conv2D_2/kernel/v
!:�2Adam/conv2D_2/bias/v
:�2Adam/bn_2/gamma/v
:�2Adam/bn_2/beta/v
$:"
�$�2Adam/fc_0/kernel/v
:�2Adam/fc_0/bias/v
#:!	�2Adam/fc_1/kernel/v
:2Adam/fc_1/bias/v
�2�
 __inference__wrapped_model_34488�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *.�+
)�&
input_1���������@@
�2�
F__inference_CatDogModel_layer_call_and_return_conditional_losses_35467
F__inference_CatDogModel_layer_call_and_return_conditional_losses_35534
F__inference_CatDogModel_layer_call_and_return_conditional_losses_36019
F__inference_CatDogModel_layer_call_and_return_conditional_losses_36109�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
+__inference_CatDogModel_layer_call_fn_35767
+__inference_CatDogModel_layer_call_fn_36158
+__inference_CatDogModel_layer_call_fn_36207
+__inference_CatDogModel_layer_call_fn_35651�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
C__inference_conv2D_0_layer_call_and_return_conditional_losses_34499�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������
�2�
(__inference_conv2D_0_layer_call_fn_34509�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������
�2�
?__inference_bn_0_layer_call_and_return_conditional_losses_36343
?__inference_bn_0_layer_call_and_return_conditional_losses_36268
?__inference_bn_0_layer_call_and_return_conditional_losses_36250
?__inference_bn_0_layer_call_and_return_conditional_losses_36325�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
$__inference_bn_0_layer_call_fn_36369
$__inference_bn_0_layer_call_fn_36281
$__inference_bn_0_layer_call_fn_36356
$__inference_bn_0_layer_call_fn_36294�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
E__inference_activation_layer_call_and_return_conditional_losses_36374�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_activation_layer_call_fn_36379�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_max_pool_0_layer_call_and_return_conditional_losses_34641�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
*__inference_max_pool_0_layer_call_fn_34647�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
B__inference_dropout_layer_call_and_return_conditional_losses_36396
B__inference_dropout_layer_call_and_return_conditional_losses_36391�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
'__inference_dropout_layer_call_fn_36401
'__inference_dropout_layer_call_fn_36406�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
C__inference_conv2D_1_layer_call_and_return_conditional_losses_34658�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+��������������������������� 
�2�
(__inference_conv2D_1_layer_call_fn_34668�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+��������������������������� 
�2�
?__inference_bn_1_layer_call_and_return_conditional_losses_36449
?__inference_bn_1_layer_call_and_return_conditional_losses_36542
?__inference_bn_1_layer_call_and_return_conditional_losses_36467
?__inference_bn_1_layer_call_and_return_conditional_losses_36524�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
$__inference_bn_1_layer_call_fn_36555
$__inference_bn_1_layer_call_fn_36493
$__inference_bn_1_layer_call_fn_36480
$__inference_bn_1_layer_call_fn_36568�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
G__inference_activation_1_layer_call_and_return_conditional_losses_36573�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
,__inference_activation_1_layer_call_fn_36578�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_max_pool_1_layer_call_and_return_conditional_losses_34800�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
*__inference_max_pool_1_layer_call_fn_34806�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
D__inference_dropout_1_layer_call_and_return_conditional_losses_36595
D__inference_dropout_1_layer_call_and_return_conditional_losses_36590�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
)__inference_dropout_1_layer_call_fn_36600
)__inference_dropout_1_layer_call_fn_36605�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
C__inference_conv2D_2_layer_call_and_return_conditional_losses_34817�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������@
�2�
(__inference_conv2D_2_layer_call_fn_34827�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������@
�2�
?__inference_bn_2_layer_call_and_return_conditional_losses_36666
?__inference_bn_2_layer_call_and_return_conditional_losses_36741
?__inference_bn_2_layer_call_and_return_conditional_losses_36648
?__inference_bn_2_layer_call_and_return_conditional_losses_36723�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
$__inference_bn_2_layer_call_fn_36754
$__inference_bn_2_layer_call_fn_36767
$__inference_bn_2_layer_call_fn_36692
$__inference_bn_2_layer_call_fn_36679�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
G__inference_activation_2_layer_call_and_return_conditional_losses_36772�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
,__inference_activation_2_layer_call_fn_36777�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_max_pool_2_layer_call_and_return_conditional_losses_34959�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
*__inference_max_pool_2_layer_call_fn_34965�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
D__inference_dropout_2_layer_call_and_return_conditional_losses_36794
D__inference_dropout_2_layer_call_and_return_conditional_losses_36789�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
)__inference_dropout_2_layer_call_fn_36804
)__inference_dropout_2_layer_call_fn_36799�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
B__inference_flatten_layer_call_and_return_conditional_losses_36810�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_flatten_layer_call_fn_36815�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
?__inference_fc_0_layer_call_and_return_conditional_losses_36826�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
$__inference_fc_0_layer_call_fn_36835�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_dropout_3_layer_call_and_return_conditional_losses_36852
D__inference_dropout_3_layer_call_and_return_conditional_losses_36847�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
)__inference_dropout_3_layer_call_fn_36857
)__inference_dropout_3_layer_call_fn_36862�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
?__inference_fc_1_layer_call_and_return_conditional_losses_36873�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
$__inference_fc_1_layer_call_fn_36882�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
2B0
#__inference_signature_wrapper_35862input_1�
F__inference_CatDogModel_layer_call_and_return_conditional_losses_35467�"#$%67=>?@QRXYZ[pqz{@�=
6�3
)�&
input_1���������@@
p

 
� "%�"
�
0���������
� �
F__inference_CatDogModel_layer_call_and_return_conditional_losses_35534�"#$%67=>?@QRXYZ[pqz{@�=
6�3
)�&
input_1���������@@
p 

 
� "%�"
�
0���������
� �
F__inference_CatDogModel_layer_call_and_return_conditional_losses_36019�"#$%67=>?@QRXYZ[pqz{?�<
5�2
(�%
inputs���������@@
p

 
� "%�"
�
0���������
� �
F__inference_CatDogModel_layer_call_and_return_conditional_losses_36109�"#$%67=>?@QRXYZ[pqz{?�<
5�2
(�%
inputs���������@@
p 

 
� "%�"
�
0���������
� �
+__inference_CatDogModel_layer_call_fn_35651t"#$%67=>?@QRXYZ[pqz{@�=
6�3
)�&
input_1���������@@
p

 
� "�����������
+__inference_CatDogModel_layer_call_fn_35767t"#$%67=>?@QRXYZ[pqz{@�=
6�3
)�&
input_1���������@@
p 

 
� "�����������
+__inference_CatDogModel_layer_call_fn_36158s"#$%67=>?@QRXYZ[pqz{?�<
5�2
(�%
inputs���������@@
p

 
� "�����������
+__inference_CatDogModel_layer_call_fn_36207s"#$%67=>?@QRXYZ[pqz{?�<
5�2
(�%
inputs���������@@
p 

 
� "�����������
 __inference__wrapped_model_34488"#$%67=>?@QRXYZ[pqz{8�5
.�+
)�&
input_1���������@@
� "+�(
&
fc_1�
fc_1����������
G__inference_activation_1_layer_call_and_return_conditional_losses_36573h7�4
-�*
(�%
inputs���������@
� "-�*
#� 
0���������@
� �
,__inference_activation_1_layer_call_fn_36578[7�4
-�*
(�%
inputs���������@
� " ����������@�
G__inference_activation_2_layer_call_and_return_conditional_losses_36772j8�5
.�+
)�&
inputs����������
� ".�+
$�!
0����������
� �
,__inference_activation_2_layer_call_fn_36777]8�5
.�+
)�&
inputs����������
� "!������������
E__inference_activation_layer_call_and_return_conditional_losses_36374h7�4
-�*
(�%
inputs���������>> 
� "-�*
#� 
0���������>> 
� �
*__inference_activation_layer_call_fn_36379[7�4
-�*
(�%
inputs���������>> 
� " ����������>> �
?__inference_bn_0_layer_call_and_return_conditional_losses_36250r"#$%;�8
1�.
(�%
inputs���������>> 
p
� "-�*
#� 
0���������>> 
� �
?__inference_bn_0_layer_call_and_return_conditional_losses_36268r"#$%;�8
1�.
(�%
inputs���������>> 
p 
� "-�*
#� 
0���������>> 
� �
?__inference_bn_0_layer_call_and_return_conditional_losses_36325�"#$%M�J
C�@
:�7
inputs+��������������������������� 
p
� "?�<
5�2
0+��������������������������� 
� �
?__inference_bn_0_layer_call_and_return_conditional_losses_36343�"#$%M�J
C�@
:�7
inputs+��������������������������� 
p 
� "?�<
5�2
0+��������������������������� 
� �
$__inference_bn_0_layer_call_fn_36281e"#$%;�8
1�.
(�%
inputs���������>> 
p
� " ����������>> �
$__inference_bn_0_layer_call_fn_36294e"#$%;�8
1�.
(�%
inputs���������>> 
p 
� " ����������>> �
$__inference_bn_0_layer_call_fn_36356�"#$%M�J
C�@
:�7
inputs+��������������������������� 
p
� "2�/+��������������������������� �
$__inference_bn_0_layer_call_fn_36369�"#$%M�J
C�@
:�7
inputs+��������������������������� 
p 
� "2�/+��������������������������� �
?__inference_bn_1_layer_call_and_return_conditional_losses_36449�=>?@M�J
C�@
:�7
inputs+���������������������������@
p
� "?�<
5�2
0+���������������������������@
� �
?__inference_bn_1_layer_call_and_return_conditional_losses_36467�=>?@M�J
C�@
:�7
inputs+���������������������������@
p 
� "?�<
5�2
0+���������������������������@
� �
?__inference_bn_1_layer_call_and_return_conditional_losses_36524r=>?@;�8
1�.
(�%
inputs���������@
p
� "-�*
#� 
0���������@
� �
?__inference_bn_1_layer_call_and_return_conditional_losses_36542r=>?@;�8
1�.
(�%
inputs���������@
p 
� "-�*
#� 
0���������@
� �
$__inference_bn_1_layer_call_fn_36480�=>?@M�J
C�@
:�7
inputs+���������������������������@
p
� "2�/+���������������������������@�
$__inference_bn_1_layer_call_fn_36493�=>?@M�J
C�@
:�7
inputs+���������������������������@
p 
� "2�/+���������������������������@�
$__inference_bn_1_layer_call_fn_36555e=>?@;�8
1�.
(�%
inputs���������@
p
� " ����������@�
$__inference_bn_1_layer_call_fn_36568e=>?@;�8
1�.
(�%
inputs���������@
p 
� " ����������@�
?__inference_bn_2_layer_call_and_return_conditional_losses_36648�XYZ[N�K
D�A
;�8
inputs,����������������������������
p
� "@�=
6�3
0,����������������������������
� �
?__inference_bn_2_layer_call_and_return_conditional_losses_36666�XYZ[N�K
D�A
;�8
inputs,����������������������������
p 
� "@�=
6�3
0,����������������������������
� �
?__inference_bn_2_layer_call_and_return_conditional_losses_36723tXYZ[<�9
2�/
)�&
inputs����������
p
� ".�+
$�!
0����������
� �
?__inference_bn_2_layer_call_and_return_conditional_losses_36741tXYZ[<�9
2�/
)�&
inputs����������
p 
� ".�+
$�!
0����������
� �
$__inference_bn_2_layer_call_fn_36679�XYZ[N�K
D�A
;�8
inputs,����������������������������
p
� "3�0,�����������������������������
$__inference_bn_2_layer_call_fn_36692�XYZ[N�K
D�A
;�8
inputs,����������������������������
p 
� "3�0,�����������������������������
$__inference_bn_2_layer_call_fn_36754gXYZ[<�9
2�/
)�&
inputs����������
p
� "!������������
$__inference_bn_2_layer_call_fn_36767gXYZ[<�9
2�/
)�&
inputs����������
p 
� "!������������
C__inference_conv2D_0_layer_call_and_return_conditional_losses_34499�I�F
?�<
:�7
inputs+���������������������������
� "?�<
5�2
0+��������������������������� 
� �
(__inference_conv2D_0_layer_call_fn_34509�I�F
?�<
:�7
inputs+���������������������������
� "2�/+��������������������������� �
C__inference_conv2D_1_layer_call_and_return_conditional_losses_34658�67I�F
?�<
:�7
inputs+��������������������������� 
� "?�<
5�2
0+���������������������������@
� �
(__inference_conv2D_1_layer_call_fn_34668�67I�F
?�<
:�7
inputs+��������������������������� 
� "2�/+���������������������������@�
C__inference_conv2D_2_layer_call_and_return_conditional_losses_34817�QRI�F
?�<
:�7
inputs+���������������������������@
� "@�=
6�3
0,����������������������������
� �
(__inference_conv2D_2_layer_call_fn_34827�QRI�F
?�<
:�7
inputs+���������������������������@
� "3�0,�����������������������������
D__inference_dropout_1_layer_call_and_return_conditional_losses_36590l;�8
1�.
(�%
inputs���������@
p
� "-�*
#� 
0���������@
� �
D__inference_dropout_1_layer_call_and_return_conditional_losses_36595l;�8
1�.
(�%
inputs���������@
p 
� "-�*
#� 
0���������@
� �
)__inference_dropout_1_layer_call_fn_36600_;�8
1�.
(�%
inputs���������@
p
� " ����������@�
)__inference_dropout_1_layer_call_fn_36605_;�8
1�.
(�%
inputs���������@
p 
� " ����������@�
D__inference_dropout_2_layer_call_and_return_conditional_losses_36789n<�9
2�/
)�&
inputs����������
p
� ".�+
$�!
0����������
� �
D__inference_dropout_2_layer_call_and_return_conditional_losses_36794n<�9
2�/
)�&
inputs����������
p 
� ".�+
$�!
0����������
� �
)__inference_dropout_2_layer_call_fn_36799a<�9
2�/
)�&
inputs����������
p
� "!������������
)__inference_dropout_2_layer_call_fn_36804a<�9
2�/
)�&
inputs����������
p 
� "!������������
D__inference_dropout_3_layer_call_and_return_conditional_losses_36847^4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� �
D__inference_dropout_3_layer_call_and_return_conditional_losses_36852^4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� ~
)__inference_dropout_3_layer_call_fn_36857Q4�1
*�'
!�
inputs����������
p
� "�����������~
)__inference_dropout_3_layer_call_fn_36862Q4�1
*�'
!�
inputs����������
p 
� "������������
B__inference_dropout_layer_call_and_return_conditional_losses_36391l;�8
1�.
(�%
inputs��������� 
p
� "-�*
#� 
0��������� 
� �
B__inference_dropout_layer_call_and_return_conditional_losses_36396l;�8
1�.
(�%
inputs��������� 
p 
� "-�*
#� 
0��������� 
� �
'__inference_dropout_layer_call_fn_36401_;�8
1�.
(�%
inputs��������� 
p
� " ���������� �
'__inference_dropout_layer_call_fn_36406_;�8
1�.
(�%
inputs��������� 
p 
� " ���������� �
?__inference_fc_0_layer_call_and_return_conditional_losses_36826^pq0�-
&�#
!�
inputs����������$
� "&�#
�
0����������
� y
$__inference_fc_0_layer_call_fn_36835Qpq0�-
&�#
!�
inputs����������$
� "������������
?__inference_fc_1_layer_call_and_return_conditional_losses_36873]z{0�-
&�#
!�
inputs����������
� "%�"
�
0���������
� x
$__inference_fc_1_layer_call_fn_36882Pz{0�-
&�#
!�
inputs����������
� "�����������
B__inference_flatten_layer_call_and_return_conditional_losses_36810b8�5
.�+
)�&
inputs����������
� "&�#
�
0����������$
� �
'__inference_flatten_layer_call_fn_36815U8�5
.�+
)�&
inputs����������
� "�����������$�
E__inference_max_pool_0_layer_call_and_return_conditional_losses_34641�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
*__inference_max_pool_0_layer_call_fn_34647�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
E__inference_max_pool_1_layer_call_and_return_conditional_losses_34800�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
*__inference_max_pool_1_layer_call_fn_34806�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
E__inference_max_pool_2_layer_call_and_return_conditional_losses_34959�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
*__inference_max_pool_2_layer_call_fn_34965�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
#__inference_signature_wrapper_35862�"#$%67=>?@QRXYZ[pqz{C�@
� 
9�6
4
input_1)�&
input_1���������@@"+�(
&
fc_1�
fc_1���������