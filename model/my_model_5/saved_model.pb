в™
™э
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
dtypetypeИ
Њ
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
executor_typestring И
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshapeИ"serve*2.2.02v2.2.0-rc4-8-g2b96f3662b8Џљ
В
conv2D_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2D_0/kernel
{
#conv2D_0/kernel/Read/ReadVariableOpReadVariableOpconv2D_0/kernel*&
_output_shapes
: *
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
А
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
В
conv2D_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @* 
shared_nameconv2D_1/kernel
{
#conv2D_1/kernel/Read/ReadVariableOpReadVariableOpconv2D_1/kernel*&
_output_shapes
: @*
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
А
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
Г
conv2D_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@А* 
shared_nameconv2D_2/kernel
|
#conv2D_2/kernel/Read/ReadVariableOpReadVariableOpconv2D_2/kernel*'
_output_shapes
:@А*
dtype0
s
conv2D_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_nameconv2D_2/bias
l
!conv2D_2/bias/Read/ReadVariableOpReadVariableOpconv2D_2/bias*
_output_shapes	
:А*
dtype0
m

bn_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_name
bn_2/gamma
f
bn_2/gamma/Read/ReadVariableOpReadVariableOp
bn_2/gamma*
_output_shapes	
:А*
dtype0
k
	bn_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_name	bn_2/beta
d
bn_2/beta/Read/ReadVariableOpReadVariableOp	bn_2/beta*
_output_shapes	
:А*
dtype0
y
bn_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*!
shared_namebn_2/moving_mean
r
$bn_2/moving_mean/Read/ReadVariableOpReadVariableOpbn_2/moving_mean*
_output_shapes	
:А*
dtype0
Б
bn_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_namebn_2/moving_variance
z
(bn_2/moving_variance/Read/ReadVariableOpReadVariableOpbn_2/moving_variance*
_output_shapes	
:А*
dtype0
o
	fc/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*
shared_name	fc/kernel
h
fc/kernel/Read/ReadVariableOpReadVariableOp	fc/kernel*
_output_shapes
:	А*
dtype0
f
fc/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	fc/bias
_
fc/bias/Read/ReadVariableOpReadVariableOpfc/bias*
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
Р
Adam/conv2D_0/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv2D_0/kernel/m
Й
*Adam/conv2D_0/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2D_0/kernel/m*&
_output_shapes
: *
dtype0
А
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
Р
Adam/conv2D_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*'
shared_nameAdam/conv2D_1/kernel/m
Й
*Adam/conv2D_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2D_1/kernel/m*&
_output_shapes
: @*
dtype0
А
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
С
Adam/conv2D_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@А*'
shared_nameAdam/conv2D_2/kernel/m
К
*Adam/conv2D_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2D_2/kernel/m*'
_output_shapes
:@А*
dtype0
Б
Adam/conv2D_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/conv2D_2/bias/m
z
(Adam/conv2D_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2D_2/bias/m*
_output_shapes	
:А*
dtype0
{
Adam/bn_2/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*"
shared_nameAdam/bn_2/gamma/m
t
%Adam/bn_2/gamma/m/Read/ReadVariableOpReadVariableOpAdam/bn_2/gamma/m*
_output_shapes	
:А*
dtype0
y
Adam/bn_2/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*!
shared_nameAdam/bn_2/beta/m
r
$Adam/bn_2/beta/m/Read/ReadVariableOpReadVariableOpAdam/bn_2/beta/m*
_output_shapes	
:А*
dtype0
}
Adam/fc/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*!
shared_nameAdam/fc/kernel/m
v
$Adam/fc/kernel/m/Read/ReadVariableOpReadVariableOpAdam/fc/kernel/m*
_output_shapes
:	А*
dtype0
t
Adam/fc/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/fc/bias/m
m
"Adam/fc/bias/m/Read/ReadVariableOpReadVariableOpAdam/fc/bias/m*
_output_shapes
:*
dtype0
Р
Adam/conv2D_0/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv2D_0/kernel/v
Й
*Adam/conv2D_0/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2D_0/kernel/v*&
_output_shapes
: *
dtype0
А
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
Р
Adam/conv2D_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*'
shared_nameAdam/conv2D_1/kernel/v
Й
*Adam/conv2D_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2D_1/kernel/v*&
_output_shapes
: @*
dtype0
А
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
С
Adam/conv2D_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@А*'
shared_nameAdam/conv2D_2/kernel/v
К
*Adam/conv2D_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2D_2/kernel/v*'
_output_shapes
:@А*
dtype0
Б
Adam/conv2D_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/conv2D_2/bias/v
z
(Adam/conv2D_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2D_2/bias/v*
_output_shapes	
:А*
dtype0
{
Adam/bn_2/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*"
shared_nameAdam/bn_2/gamma/v
t
%Adam/bn_2/gamma/v/Read/ReadVariableOpReadVariableOpAdam/bn_2/gamma/v*
_output_shapes	
:А*
dtype0
y
Adam/bn_2/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*!
shared_nameAdam/bn_2/beta/v
r
$Adam/bn_2/beta/v/Read/ReadVariableOpReadVariableOpAdam/bn_2/beta/v*
_output_shapes	
:А*
dtype0
}
Adam/fc/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*!
shared_nameAdam/fc/kernel/v
v
$Adam/fc/kernel/v/Read/ReadVariableOpReadVariableOpAdam/fc/kernel/v*
_output_shapes
:	А*
dtype0
t
Adam/fc/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/fc/bias/v
m
"Adam/fc/bias/v/Read/ReadVariableOpReadVariableOpAdam/fc/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
Ђb
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*жa
value№aBўa B“a
э
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
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
layer_with_weights-4
layer-10
layer_with_weights-5
layer-11
layer-12
layer-13
layer-14
layer_with_weights-6
layer-15
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
 
R
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
 	keras_api
Ч
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
h

2kernel
3bias
4regularization_losses
5trainable_variables
6	variables
7	keras_api
Ч
8axis
	9gamma
:beta
;moving_mean
<moving_variance
=regularization_losses
>trainable_variables
?	variables
@	keras_api
R
Aregularization_losses
Btrainable_variables
C	variables
D	keras_api
R
Eregularization_losses
Ftrainable_variables
G	variables
H	keras_api
h

Ikernel
Jbias
Kregularization_losses
Ltrainable_variables
M	variables
N	keras_api
Ч
Oaxis
	Pgamma
Qbeta
Rmoving_mean
Smoving_variance
Tregularization_losses
Utrainable_variables
V	variables
W	keras_api
R
Xregularization_losses
Ytrainable_variables
Z	variables
[	keras_api
R
\regularization_losses
]trainable_variables
^	variables
_	keras_api
R
`regularization_losses
atrainable_variables
b	variables
c	keras_api
h

dkernel
ebias
fregularization_losses
gtrainable_variables
h	variables
i	keras_api
Ў
jiter

kbeta_1

lbeta_2
	mdecay
nlearning_ratem mЋ"mћ#mЌ2mќ3mѕ9m–:m—Im“Jm”Pm‘Qm’dm÷em„vЎvў"vЏ#vџ2v№3vЁ9vё:vяIvаJvбPvвQvгdvдevе
 
f
0
1
"2
#3
24
35
96
:7
I8
J9
P10
Q11
d12
e13
Ц
0
1
"2
#3
$4
%5
26
37
98
:9
;10
<11
I12
J13
P14
Q15
R16
S17
d18
e19
≠
regularization_losses
ometrics
trainable_variables
pnon_trainable_variables
	variables

qlayers
rlayer_regularization_losses
slayer_metrics
 
 
 
 
≠
regularization_losses
tmetrics
trainable_variables
unon_trainable_variables
	variables

vlayers
wlayer_regularization_losses
xlayer_metrics
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
≠
regularization_losses
ymetrics
trainable_variables
znon_trainable_variables
	variables

{layers
|layer_regularization_losses
}layer_metrics
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
∞
&regularization_losses
~metrics
'trainable_variables
non_trainable_variables
(	variables
Аlayers
 Бlayer_regularization_losses
Вlayer_metrics
 
 
 
≤
*regularization_losses
Гmetrics
+trainable_variables
Дnon_trainable_variables
,	variables
Еlayers
 Жlayer_regularization_losses
Зlayer_metrics
 
 
 
≤
.regularization_losses
Иmetrics
/trainable_variables
Йnon_trainable_variables
0	variables
Кlayers
 Лlayer_regularization_losses
Мlayer_metrics
[Y
VARIABLE_VALUEconv2D_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2D_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

20
31

20
31
≤
4regularization_losses
Нmetrics
5trainable_variables
Оnon_trainable_variables
6	variables
Пlayers
 Рlayer_regularization_losses
Сlayer_metrics
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
90
:1

90
:1
;2
<3
≤
=regularization_losses
Тmetrics
>trainable_variables
Уnon_trainable_variables
?	variables
Фlayers
 Хlayer_regularization_losses
Цlayer_metrics
 
 
 
≤
Aregularization_losses
Чmetrics
Btrainable_variables
Шnon_trainable_variables
C	variables
Щlayers
 Ъlayer_regularization_losses
Ыlayer_metrics
 
 
 
≤
Eregularization_losses
Ьmetrics
Ftrainable_variables
Эnon_trainable_variables
G	variables
Юlayers
 Яlayer_regularization_losses
†layer_metrics
[Y
VARIABLE_VALUEconv2D_2/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2D_2/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

I0
J1

I0
J1
≤
Kregularization_losses
°metrics
Ltrainable_variables
Ґnon_trainable_variables
M	variables
£layers
 §layer_regularization_losses
•layer_metrics
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
P0
Q1

P0
Q1
R2
S3
≤
Tregularization_losses
¶metrics
Utrainable_variables
Іnon_trainable_variables
V	variables
®layers
 ©layer_regularization_losses
™layer_metrics
 
 
 
≤
Xregularization_losses
Ђmetrics
Ytrainable_variables
ђnon_trainable_variables
Z	variables
≠layers
 Ѓlayer_regularization_losses
ѓlayer_metrics
 
 
 
≤
\regularization_losses
∞metrics
]trainable_variables
±non_trainable_variables
^	variables
≤layers
 ≥layer_regularization_losses
іlayer_metrics
 
 
 
≤
`regularization_losses
µmetrics
atrainable_variables
ґnon_trainable_variables
b	variables
Јlayers
 Єlayer_regularization_losses
єlayer_metrics
US
VARIABLE_VALUE	fc/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEfc/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
 

d0
e1

d0
e1
≤
fregularization_losses
Їmetrics
gtrainable_variables
їnon_trainable_variables
h	variables
Љlayers
 љlayer_regularization_losses
Њlayer_metrics
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

њ0
ј1
*
$0
%1
;2
<3
R4
S5
v
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

;0
<1
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
R0
S1
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

Ѕtotal

¬count
√	variables
ƒ	keras_api
I

≈total

∆count
«
_fn_kwargs
»	variables
…	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

Ѕ0
¬1

√	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

≈0
∆1

»	variables
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
xv
VARIABLE_VALUEAdam/fc/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/fc/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
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
xv
VARIABLE_VALUEAdam/fc/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/fc/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
К
serving_default_input_1Placeholder*/
_output_shapes
:€€€€€€€€€@@*
dtype0*$
shape:€€€€€€€€€@@
и
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv2D_0/kernelconv2D_0/bias
bn_0/gamma	bn_0/betabn_0/moving_meanbn_0/moving_varianceconv2D_1/kernelconv2D_1/bias
bn_1/gamma	bn_1/betabn_1/moving_meanbn_1/moving_varianceconv2D_2/kernelconv2D_2/bias
bn_2/gamma	bn_2/betabn_2/moving_meanbn_2/moving_variance	fc/kernelfc/bias* 
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€*6
_read_only_resource_inputs
	
**
config_proto

GPU 

CPU2J 8*,
f'R%
#__inference_signature_wrapper_12147
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
с
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#conv2D_0/kernel/Read/ReadVariableOp!conv2D_0/bias/Read/ReadVariableOpbn_0/gamma/Read/ReadVariableOpbn_0/beta/Read/ReadVariableOp$bn_0/moving_mean/Read/ReadVariableOp(bn_0/moving_variance/Read/ReadVariableOp#conv2D_1/kernel/Read/ReadVariableOp!conv2D_1/bias/Read/ReadVariableOpbn_1/gamma/Read/ReadVariableOpbn_1/beta/Read/ReadVariableOp$bn_1/moving_mean/Read/ReadVariableOp(bn_1/moving_variance/Read/ReadVariableOp#conv2D_2/kernel/Read/ReadVariableOp!conv2D_2/bias/Read/ReadVariableOpbn_2/gamma/Read/ReadVariableOpbn_2/beta/Read/ReadVariableOp$bn_2/moving_mean/Read/ReadVariableOp(bn_2/moving_variance/Read/ReadVariableOpfc/kernel/Read/ReadVariableOpfc/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp*Adam/conv2D_0/kernel/m/Read/ReadVariableOp(Adam/conv2D_0/bias/m/Read/ReadVariableOp%Adam/bn_0/gamma/m/Read/ReadVariableOp$Adam/bn_0/beta/m/Read/ReadVariableOp*Adam/conv2D_1/kernel/m/Read/ReadVariableOp(Adam/conv2D_1/bias/m/Read/ReadVariableOp%Adam/bn_1/gamma/m/Read/ReadVariableOp$Adam/bn_1/beta/m/Read/ReadVariableOp*Adam/conv2D_2/kernel/m/Read/ReadVariableOp(Adam/conv2D_2/bias/m/Read/ReadVariableOp%Adam/bn_2/gamma/m/Read/ReadVariableOp$Adam/bn_2/beta/m/Read/ReadVariableOp$Adam/fc/kernel/m/Read/ReadVariableOp"Adam/fc/bias/m/Read/ReadVariableOp*Adam/conv2D_0/kernel/v/Read/ReadVariableOp(Adam/conv2D_0/bias/v/Read/ReadVariableOp%Adam/bn_0/gamma/v/Read/ReadVariableOp$Adam/bn_0/beta/v/Read/ReadVariableOp*Adam/conv2D_1/kernel/v/Read/ReadVariableOp(Adam/conv2D_1/bias/v/Read/ReadVariableOp%Adam/bn_1/gamma/v/Read/ReadVariableOp$Adam/bn_1/beta/v/Read/ReadVariableOp*Adam/conv2D_2/kernel/v/Read/ReadVariableOp(Adam/conv2D_2/bias/v/Read/ReadVariableOp%Adam/bn_2/gamma/v/Read/ReadVariableOp$Adam/bn_2/beta/v/Read/ReadVariableOp$Adam/fc/kernel/v/Read/ReadVariableOp"Adam/fc/bias/v/Read/ReadVariableOpConst*F
Tin?
=2;	*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*'
f"R 
__inference__traced_save_13183
ш	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2D_0/kernelconv2D_0/bias
bn_0/gamma	bn_0/betabn_0/moving_meanbn_0/moving_varianceconv2D_1/kernelconv2D_1/bias
bn_1/gamma	bn_1/betabn_1/moving_meanbn_1/moving_varianceconv2D_2/kernelconv2D_2/bias
bn_2/gamma	bn_2/betabn_2/moving_meanbn_2/moving_variance	fc/kernelfc/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/conv2D_0/kernel/mAdam/conv2D_0/bias/mAdam/bn_0/gamma/mAdam/bn_0/beta/mAdam/conv2D_1/kernel/mAdam/conv2D_1/bias/mAdam/bn_1/gamma/mAdam/bn_1/beta/mAdam/conv2D_2/kernel/mAdam/conv2D_2/bias/mAdam/bn_2/gamma/mAdam/bn_2/beta/mAdam/fc/kernel/mAdam/fc/bias/mAdam/conv2D_0/kernel/vAdam/conv2D_0/bias/vAdam/bn_0/gamma/vAdam/bn_0/beta/vAdam/conv2D_1/kernel/vAdam/conv2D_1/bias/vAdam/bn_1/gamma/vAdam/bn_1/beta/vAdam/conv2D_2/kernel/vAdam/conv2D_2/bias/vAdam/bn_2/gamma/vAdam/bn_2/beta/vAdam/fc/kernel/vAdam/fc/bias/v*E
Tin>
<2:*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8**
f%R#
!__inference__traced_restore_13366Мќ
з$
∆
?__inference_bn_2_layer_call_and_return_conditional_losses_11378

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1®
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpЃ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ќ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
Const∞
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xњ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub¶
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02 
AssignMovingAvg/ReadVariableOpя
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:А2
AssignMovingAvg/sub_1»
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:А2
AssignMovingAvg/mul«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpґ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x«
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subђ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02"
 AssignMovingAvg_1/ReadVariableOpл
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:А2
AssignMovingAvg_1/sub_1“
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:А2
AssignMovingAvg_1/mul’
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp—
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
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
ы
a
E__inference_max_pool_2_layer_call_and_return_conditional_losses_11426

inputs
identity≠
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
€
F
*__inference_max_pool_2_layer_call_fn_11432

inputs
identityƒ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_max_pool_2_layer_call_and_return_conditional_losses_114262
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ґv
і
__inference__traced_save_13183
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
/savev2_bn_2_moving_variance_read_readvariableop(
$savev2_fc_kernel_read_readvariableop&
"savev2_fc_bias_read_readvariableop(
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
+savev2_adam_bn_2_beta_m_read_readvariableop/
+savev2_adam_fc_kernel_m_read_readvariableop-
)savev2_adam_fc_bias_m_read_readvariableop5
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
+savev2_adam_bn_2_beta_v_read_readvariableop/
+savev2_adam_fc_kernel_v_read_readvariableop-
)savev2_adam_fc_bias_v_read_readvariableop
savev2_1_const

identity_1ИҐMergeV2CheckpointsҐSaveV2ҐSaveV2_1П
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
ConstН
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_4bd9b0dac2e343538eb8cf37642ce492/part2	
Const_1Л
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
ShardedFilename/shard¶
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename«
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:9*
dtype0*ў
valueѕBћ9B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_namesы
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:9*
dtype0*Е
value|Bz9B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices≥
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_conv2d_0_kernel_read_readvariableop(savev2_conv2d_0_bias_read_readvariableop%savev2_bn_0_gamma_read_readvariableop$savev2_bn_0_beta_read_readvariableop+savev2_bn_0_moving_mean_read_readvariableop/savev2_bn_0_moving_variance_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop%savev2_bn_1_gamma_read_readvariableop$savev2_bn_1_beta_read_readvariableop+savev2_bn_1_moving_mean_read_readvariableop/savev2_bn_1_moving_variance_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop%savev2_bn_2_gamma_read_readvariableop$savev2_bn_2_beta_read_readvariableop+savev2_bn_2_moving_mean_read_readvariableop/savev2_bn_2_moving_variance_read_readvariableop$savev2_fc_kernel_read_readvariableop"savev2_fc_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop1savev2_adam_conv2d_0_kernel_m_read_readvariableop/savev2_adam_conv2d_0_bias_m_read_readvariableop,savev2_adam_bn_0_gamma_m_read_readvariableop+savev2_adam_bn_0_beta_m_read_readvariableop1savev2_adam_conv2d_1_kernel_m_read_readvariableop/savev2_adam_conv2d_1_bias_m_read_readvariableop,savev2_adam_bn_1_gamma_m_read_readvariableop+savev2_adam_bn_1_beta_m_read_readvariableop1savev2_adam_conv2d_2_kernel_m_read_readvariableop/savev2_adam_conv2d_2_bias_m_read_readvariableop,savev2_adam_bn_2_gamma_m_read_readvariableop+savev2_adam_bn_2_beta_m_read_readvariableop+savev2_adam_fc_kernel_m_read_readvariableop)savev2_adam_fc_bias_m_read_readvariableop1savev2_adam_conv2d_0_kernel_v_read_readvariableop/savev2_adam_conv2d_0_bias_v_read_readvariableop,savev2_adam_bn_0_gamma_v_read_readvariableop+savev2_adam_bn_0_beta_v_read_readvariableop1savev2_adam_conv2d_1_kernel_v_read_readvariableop/savev2_adam_conv2d_1_bias_v_read_readvariableop,savev2_adam_bn_1_gamma_v_read_readvariableop+savev2_adam_bn_1_beta_v_read_readvariableop1savev2_adam_conv2d_2_kernel_v_read_readvariableop/savev2_adam_conv2d_2_bias_v_read_readvariableop,savev2_adam_bn_2_gamma_v_read_readvariableop+savev2_adam_bn_2_beta_v_read_readvariableop+savev2_adam_fc_kernel_v_read_readvariableop)savev2_adam_fc_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *G
dtypes=
;29	2
SaveV2Г
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shardђ
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1Ґ
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_namesО
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slicesѕ
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1г
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesђ
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

IdentityБ

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*‘
_input_shapes¬
њ: : : : : : : : @:@:@:@:@:@:@А:А:А:А:А:А:	А:: : : : : : : : : : : : : : @:@:@:@:@А:А:А:А:	А:: : : : : @:@:@:@:@А:А:А:А:	А:: 2(
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
: : 
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
: @: 
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
:@А:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:%!

_output_shapes
:	А: 

_output_shapes
::
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
: :,(
&
_output_shapes
: : 

_output_shapes
: :  

_output_shapes
: : !

_output_shapes
: :,"(
&
_output_shapes
: @: #

_output_shapes
:@: $

_output_shapes
:@: %

_output_shapes
:@:-&)
'
_output_shapes
:@А:!'

_output_shapes	
:А:!(

_output_shapes	
:А:!)

_output_shapes	
:А:%*!

_output_shapes
:	А: +

_output_shapes
::,,(
&
_output_shapes
: : -

_output_shapes
: : .

_output_shapes
: : /

_output_shapes
: :,0(
&
_output_shapes
: @: 1

_output_shapes
:@: 2

_output_shapes
:@: 3

_output_shapes
:@:-4)
'
_output_shapes
:@А:!5

_output_shapes	
:А:!6

_output_shapes	
:А:!7

_output_shapes	
:А:%8!

_output_shapes
:	А: 9

_output_shapes
:::

_output_shapes
: 
Н$
∆
?__inference_bn_0_layer_call_and_return_conditional_losses_12556

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ј
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€@@ : : : : :*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
Const∞
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xњ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub•
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpё
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub_1«
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/mul«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpґ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x«
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЂ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpк
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub_1—
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/mul’
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpЊ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*/
_output_shapes
:€€€€€€€€€@@ 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€@@ ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:€€€€€€€€€@@ 
 
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
ї	
Ђ
C__inference_conv2D_0_layer_call_and_return_conditional_losses_10966

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpґ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ *
paddingVALID*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpЪ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2	
BiasAdd~
IdentityIdentityBiasAdd:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:::i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Љ
^
B__inference_flatten_layer_call_and_return_conditional_losses_12960

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€А  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€А:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
’
Ч
$__inference_bn_2_layer_call_fn_12869

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*&
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*H
fCRA
?__inference_bn_2_layer_call_and_return_conditional_losses_114092
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
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
Л
ш
?__inference_bn_2_layer_call_and_return_conditional_losses_11409

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1®
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpЃ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1б
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3Г
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:::::j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
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
ў
c
G__inference_activation_2_layer_call_and_return_conditional_losses_11737

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:€€€€€€€€€

А2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:€€€€€€€€€

А2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€

А:X T
0
_output_shapes
:€€€€€€€€€

А
 
_user_specified_nameinputs
ё
}
(__inference_conv2D_2_layer_call_fn_11294

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallм
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_conv2D_2_layer_call_and_return_conditional_losses_112842
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ј	
Ђ
C__inference_conv2D_2_layer_call_and_return_conditional_losses_11284

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИЦ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype02
Conv2D/ReadVariableOpЈ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
paddingVALID*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЫ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2	
BiasAdd
IdentityIdentityBiasAdd:output:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:::i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Ъ
H
,__inference_activation_2_layer_call_fn_12954

inputs
identityђ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*0
_output_shapes
:€€€€€€€€€

А* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*P
fKRI
G__inference_activation_2_layer_call_and_return_conditional_losses_117372
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:€€€€€€€€€

А2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€

А:X T
0
_output_shapes
:€€€€€€€€€

А
 
_user_specified_nameinputs
щ
Ч
+__inference_CatDogModel_layer_call_fn_12438

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

unknown_18
identityИҐStatefulPartitionedCall«
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€*6
_read_only_resource_inputs
	
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_CatDogModel_layer_call_and_return_conditional_losses_120132
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:€€€€€€€€€@@::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€@@
 
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
: 
Зс
‘
!__inference__traced_restore_13366
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
(assignvariableop_17_bn_2_moving_variance!
assignvariableop_18_fc_kernel
assignvariableop_19_fc_bias!
assignvariableop_20_adam_iter#
assignvariableop_21_adam_beta_1#
assignvariableop_22_adam_beta_2"
assignvariableop_23_adam_decay*
&assignvariableop_24_adam_learning_rate
assignvariableop_25_total
assignvariableop_26_count
assignvariableop_27_total_1
assignvariableop_28_count_1.
*assignvariableop_29_adam_conv2d_0_kernel_m,
(assignvariableop_30_adam_conv2d_0_bias_m)
%assignvariableop_31_adam_bn_0_gamma_m(
$assignvariableop_32_adam_bn_0_beta_m.
*assignvariableop_33_adam_conv2d_1_kernel_m,
(assignvariableop_34_adam_conv2d_1_bias_m)
%assignvariableop_35_adam_bn_1_gamma_m(
$assignvariableop_36_adam_bn_1_beta_m.
*assignvariableop_37_adam_conv2d_2_kernel_m,
(assignvariableop_38_adam_conv2d_2_bias_m)
%assignvariableop_39_adam_bn_2_gamma_m(
$assignvariableop_40_adam_bn_2_beta_m(
$assignvariableop_41_adam_fc_kernel_m&
"assignvariableop_42_adam_fc_bias_m.
*assignvariableop_43_adam_conv2d_0_kernel_v,
(assignvariableop_44_adam_conv2d_0_bias_v)
%assignvariableop_45_adam_bn_0_gamma_v(
$assignvariableop_46_adam_bn_0_beta_v.
*assignvariableop_47_adam_conv2d_1_kernel_v,
(assignvariableop_48_adam_conv2d_1_bias_v)
%assignvariableop_49_adam_bn_1_gamma_v(
$assignvariableop_50_adam_bn_1_beta_v.
*assignvariableop_51_adam_conv2d_2_kernel_v,
(assignvariableop_52_adam_conv2d_2_bias_v)
%assignvariableop_53_adam_bn_2_gamma_v(
$assignvariableop_54_adam_bn_2_beta_v(
$assignvariableop_55_adam_fc_kernel_v&
"assignvariableop_56_adam_fc_bias_v
identity_58ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_10ҐAssignVariableOp_11ҐAssignVariableOp_12ҐAssignVariableOp_13ҐAssignVariableOp_14ҐAssignVariableOp_15ҐAssignVariableOp_16ҐAssignVariableOp_17ҐAssignVariableOp_18ҐAssignVariableOp_19ҐAssignVariableOp_2ҐAssignVariableOp_20ҐAssignVariableOp_21ҐAssignVariableOp_22ҐAssignVariableOp_23ҐAssignVariableOp_24ҐAssignVariableOp_25ҐAssignVariableOp_26ҐAssignVariableOp_27ҐAssignVariableOp_28ҐAssignVariableOp_29ҐAssignVariableOp_3ҐAssignVariableOp_30ҐAssignVariableOp_31ҐAssignVariableOp_32ҐAssignVariableOp_33ҐAssignVariableOp_34ҐAssignVariableOp_35ҐAssignVariableOp_36ҐAssignVariableOp_37ҐAssignVariableOp_38ҐAssignVariableOp_39ҐAssignVariableOp_4ҐAssignVariableOp_40ҐAssignVariableOp_41ҐAssignVariableOp_42ҐAssignVariableOp_43ҐAssignVariableOp_44ҐAssignVariableOp_45ҐAssignVariableOp_46ҐAssignVariableOp_47ҐAssignVariableOp_48ҐAssignVariableOp_49ҐAssignVariableOp_5ҐAssignVariableOp_50ҐAssignVariableOp_51ҐAssignVariableOp_52ҐAssignVariableOp_53ҐAssignVariableOp_54ҐAssignVariableOp_55ҐAssignVariableOp_56ҐAssignVariableOp_6ҐAssignVariableOp_7ҐAssignVariableOp_8ҐAssignVariableOp_9Ґ	RestoreV2ҐRestoreV2_1Ќ
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:9*
dtype0*ў
valueѕBћ9B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_namesБ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:9*
dtype0*Е
value|Bz9B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesЋ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*ъ
_output_shapesз
д:::::::::::::::::::::::::::::::::::::::::::::::::::::::::*G
dtypes=
;29	2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

IdentityР
AssignVariableOpAssignVariableOp assignvariableop_conv2d_0_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1Ц
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv2d_0_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2У
AssignVariableOp_2AssignVariableOpassignvariableop_2_bn_0_gammaIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3Т
AssignVariableOp_3AssignVariableOpassignvariableop_3_bn_0_betaIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4Щ
AssignVariableOp_4AssignVariableOp#assignvariableop_4_bn_0_moving_meanIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5Э
AssignVariableOp_5AssignVariableOp'assignvariableop_5_bn_0_moving_varianceIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6Ш
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv2d_1_kernelIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7Ц
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv2d_1_biasIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8У
AssignVariableOp_8AssignVariableOpassignvariableop_8_bn_1_gammaIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9Т
AssignVariableOp_9AssignVariableOpassignvariableop_9_bn_1_betaIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10Э
AssignVariableOp_10AssignVariableOp$assignvariableop_10_bn_1_moving_meanIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11°
AssignVariableOp_11AssignVariableOp(assignvariableop_11_bn_1_moving_varianceIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12Ь
AssignVariableOp_12AssignVariableOp#assignvariableop_12_conv2d_2_kernelIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13Ъ
AssignVariableOp_13AssignVariableOp!assignvariableop_13_conv2d_2_biasIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14Ч
AssignVariableOp_14AssignVariableOpassignvariableop_14_bn_2_gammaIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15Ц
AssignVariableOp_15AssignVariableOpassignvariableop_15_bn_2_betaIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16Э
AssignVariableOp_16AssignVariableOp$assignvariableop_16_bn_2_moving_meanIdentity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17°
AssignVariableOp_17AssignVariableOp(assignvariableop_17_bn_2_moving_varianceIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18Ц
AssignVariableOp_18AssignVariableOpassignvariableop_18_fc_kernelIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19Ф
AssignVariableOp_19AssignVariableOpassignvariableop_19_fc_biasIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0	*
_output_shapes
:2
Identity_20Ц
AssignVariableOp_20AssignVariableOpassignvariableop_20_adam_iterIdentity_20:output:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21Ш
AssignVariableOp_21AssignVariableOpassignvariableop_21_adam_beta_1Identity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:2
Identity_22Ш
AssignVariableOp_22AssignVariableOpassignvariableop_22_adam_beta_2Identity_22:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_22_
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:2
Identity_23Ч
AssignVariableOp_23AssignVariableOpassignvariableop_23_adam_decayIdentity_23:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_23_
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:2
Identity_24Я
AssignVariableOp_24AssignVariableOp&assignvariableop_24_adam_learning_rateIdentity_24:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_24_
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:2
Identity_25Т
AssignVariableOp_25AssignVariableOpassignvariableop_25_totalIdentity_25:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_25_
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:2
Identity_26Т
AssignVariableOp_26AssignVariableOpassignvariableop_26_countIdentity_26:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_26_
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:2
Identity_27Ф
AssignVariableOp_27AssignVariableOpassignvariableop_27_total_1Identity_27:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_27_
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:2
Identity_28Ф
AssignVariableOp_28AssignVariableOpassignvariableop_28_count_1Identity_28:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_28_
Identity_29IdentityRestoreV2:tensors:29*
T0*
_output_shapes
:2
Identity_29£
AssignVariableOp_29AssignVariableOp*assignvariableop_29_adam_conv2d_0_kernel_mIdentity_29:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_29_
Identity_30IdentityRestoreV2:tensors:30*
T0*
_output_shapes
:2
Identity_30°
AssignVariableOp_30AssignVariableOp(assignvariableop_30_adam_conv2d_0_bias_mIdentity_30:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_30_
Identity_31IdentityRestoreV2:tensors:31*
T0*
_output_shapes
:2
Identity_31Ю
AssignVariableOp_31AssignVariableOp%assignvariableop_31_adam_bn_0_gamma_mIdentity_31:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_31_
Identity_32IdentityRestoreV2:tensors:32*
T0*
_output_shapes
:2
Identity_32Э
AssignVariableOp_32AssignVariableOp$assignvariableop_32_adam_bn_0_beta_mIdentity_32:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_32_
Identity_33IdentityRestoreV2:tensors:33*
T0*
_output_shapes
:2
Identity_33£
AssignVariableOp_33AssignVariableOp*assignvariableop_33_adam_conv2d_1_kernel_mIdentity_33:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_33_
Identity_34IdentityRestoreV2:tensors:34*
T0*
_output_shapes
:2
Identity_34°
AssignVariableOp_34AssignVariableOp(assignvariableop_34_adam_conv2d_1_bias_mIdentity_34:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_34_
Identity_35IdentityRestoreV2:tensors:35*
T0*
_output_shapes
:2
Identity_35Ю
AssignVariableOp_35AssignVariableOp%assignvariableop_35_adam_bn_1_gamma_mIdentity_35:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_35_
Identity_36IdentityRestoreV2:tensors:36*
T0*
_output_shapes
:2
Identity_36Э
AssignVariableOp_36AssignVariableOp$assignvariableop_36_adam_bn_1_beta_mIdentity_36:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_36_
Identity_37IdentityRestoreV2:tensors:37*
T0*
_output_shapes
:2
Identity_37£
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_conv2d_2_kernel_mIdentity_37:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_37_
Identity_38IdentityRestoreV2:tensors:38*
T0*
_output_shapes
:2
Identity_38°
AssignVariableOp_38AssignVariableOp(assignvariableop_38_adam_conv2d_2_bias_mIdentity_38:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_38_
Identity_39IdentityRestoreV2:tensors:39*
T0*
_output_shapes
:2
Identity_39Ю
AssignVariableOp_39AssignVariableOp%assignvariableop_39_adam_bn_2_gamma_mIdentity_39:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_39_
Identity_40IdentityRestoreV2:tensors:40*
T0*
_output_shapes
:2
Identity_40Э
AssignVariableOp_40AssignVariableOp$assignvariableop_40_adam_bn_2_beta_mIdentity_40:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_40_
Identity_41IdentityRestoreV2:tensors:41*
T0*
_output_shapes
:2
Identity_41Э
AssignVariableOp_41AssignVariableOp$assignvariableop_41_adam_fc_kernel_mIdentity_41:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_41_
Identity_42IdentityRestoreV2:tensors:42*
T0*
_output_shapes
:2
Identity_42Ы
AssignVariableOp_42AssignVariableOp"assignvariableop_42_adam_fc_bias_mIdentity_42:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_42_
Identity_43IdentityRestoreV2:tensors:43*
T0*
_output_shapes
:2
Identity_43£
AssignVariableOp_43AssignVariableOp*assignvariableop_43_adam_conv2d_0_kernel_vIdentity_43:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_43_
Identity_44IdentityRestoreV2:tensors:44*
T0*
_output_shapes
:2
Identity_44°
AssignVariableOp_44AssignVariableOp(assignvariableop_44_adam_conv2d_0_bias_vIdentity_44:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_44_
Identity_45IdentityRestoreV2:tensors:45*
T0*
_output_shapes
:2
Identity_45Ю
AssignVariableOp_45AssignVariableOp%assignvariableop_45_adam_bn_0_gamma_vIdentity_45:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_45_
Identity_46IdentityRestoreV2:tensors:46*
T0*
_output_shapes
:2
Identity_46Э
AssignVariableOp_46AssignVariableOp$assignvariableop_46_adam_bn_0_beta_vIdentity_46:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_46_
Identity_47IdentityRestoreV2:tensors:47*
T0*
_output_shapes
:2
Identity_47£
AssignVariableOp_47AssignVariableOp*assignvariableop_47_adam_conv2d_1_kernel_vIdentity_47:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_47_
Identity_48IdentityRestoreV2:tensors:48*
T0*
_output_shapes
:2
Identity_48°
AssignVariableOp_48AssignVariableOp(assignvariableop_48_adam_conv2d_1_bias_vIdentity_48:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_48_
Identity_49IdentityRestoreV2:tensors:49*
T0*
_output_shapes
:2
Identity_49Ю
AssignVariableOp_49AssignVariableOp%assignvariableop_49_adam_bn_1_gamma_vIdentity_49:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_49_
Identity_50IdentityRestoreV2:tensors:50*
T0*
_output_shapes
:2
Identity_50Э
AssignVariableOp_50AssignVariableOp$assignvariableop_50_adam_bn_1_beta_vIdentity_50:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_50_
Identity_51IdentityRestoreV2:tensors:51*
T0*
_output_shapes
:2
Identity_51£
AssignVariableOp_51AssignVariableOp*assignvariableop_51_adam_conv2d_2_kernel_vIdentity_51:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_51_
Identity_52IdentityRestoreV2:tensors:52*
T0*
_output_shapes
:2
Identity_52°
AssignVariableOp_52AssignVariableOp(assignvariableop_52_adam_conv2d_2_bias_vIdentity_52:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_52_
Identity_53IdentityRestoreV2:tensors:53*
T0*
_output_shapes
:2
Identity_53Ю
AssignVariableOp_53AssignVariableOp%assignvariableop_53_adam_bn_2_gamma_vIdentity_53:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_53_
Identity_54IdentityRestoreV2:tensors:54*
T0*
_output_shapes
:2
Identity_54Э
AssignVariableOp_54AssignVariableOp$assignvariableop_54_adam_bn_2_beta_vIdentity_54:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_54_
Identity_55IdentityRestoreV2:tensors:55*
T0*
_output_shapes
:2
Identity_55Э
AssignVariableOp_55AssignVariableOp$assignvariableop_55_adam_fc_kernel_vIdentity_55:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_55_
Identity_56IdentityRestoreV2:tensors:56*
T0*
_output_shapes
:2
Identity_56Ы
AssignVariableOp_56AssignVariableOp"assignvariableop_56_adam_fc_bias_vIdentity_56:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_56®
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_namesФ
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slicesƒ
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
NoOpƒ

Identity_57Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_57—

Identity_58IdentityIdentity_57:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_58"#
identity_58Identity_58:output:0*ы
_input_shapesй
ж: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_56AssignVariableOp_562(
AssignVariableOp_6AssignVariableOp_62(
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
: 
з$
∆
?__inference_bn_2_layer_call_and_return_conditional_losses_12825

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1®
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpЃ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ќ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
Const∞
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xњ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub¶
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02 
AssignMovingAvg/ReadVariableOpя
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:А2
AssignMovingAvg/sub_1»
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:А2
AssignMovingAvg/mul«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpґ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x«
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subђ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02"
 AssignMovingAvg_1/ReadVariableOpл
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:А2
AssignMovingAvg_1/sub_1“
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:А2
AssignMovingAvg_1/mul’
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp—
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
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
ґ
ш
?__inference_bn_0_layer_call_and_return_conditional_losses_11490

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1 
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€@@ : : : : :*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€@@ 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€@@ :::::W S
/
_output_shapes
:€€€€€€€€€@@ 
 
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
√J
ѕ
F__inference_CatDogModel_layer_call_and_return_conditional_losses_12348

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
/bn_2_fusedbatchnormv3_readvariableop_1_resource%
!fc_matmul_readvariableop_resource&
"fc_biasadd_readvariableop_resource
identityИЂ
zero_padding2d/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             2
zero_padding2d/Pad/paddingsЧ
zero_padding2d/PadPadinputs$zero_padding2d/Pad/paddings:output:0*
T0*/
_output_shapes
:€€€€€€€€€DD2
zero_padding2d/Pad∞
conv2D_0/Conv2D/ReadVariableOpReadVariableOp'conv2d_0_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2D_0/Conv2D/ReadVariableOp‘
conv2D_0/Conv2DConv2Dzero_padding2d/Pad:output:0&conv2D_0/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@@ *
paddingVALID*
strides
2
conv2D_0/Conv2DІ
conv2D_0/BiasAdd/ReadVariableOpReadVariableOp(conv2d_0_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2D_0/BiasAdd/ReadVariableOpђ
conv2D_0/BiasAddBiasAddconv2D_0/Conv2D:output:0'conv2D_0/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@@ 2
conv2D_0/BiasAddГ
bn_0/ReadVariableOpReadVariableOpbn_0_readvariableop_resource*
_output_shapes
: *
dtype02
bn_0/ReadVariableOpЙ
bn_0/ReadVariableOp_1ReadVariableOpbn_0_readvariableop_1_resource*
_output_shapes
: *
dtype02
bn_0/ReadVariableOp_1ґ
$bn_0/FusedBatchNormV3/ReadVariableOpReadVariableOp-bn_0_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02&
$bn_0/FusedBatchNormV3/ReadVariableOpЉ
&bn_0/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp/bn_0_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&bn_0/FusedBatchNormV3/ReadVariableOp_1ы
bn_0/FusedBatchNormV3FusedBatchNormV3conv2D_0/BiasAdd:output:0bn_0/ReadVariableOp:value:0bn_0/ReadVariableOp_1:value:0,bn_0/FusedBatchNormV3/ReadVariableOp:value:0.bn_0/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€@@ : : : : :*
epsilon%oГ:*
is_training( 2
bn_0/FusedBatchNormV3
activation/ReluRelubn_0/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€@@ 2
activation/Reluњ
max_pool_0/MaxPoolMaxPoolactivation/Relu:activations:0*/
_output_shapes
:€€€€€€€€€   *
ksize
*
paddingVALID*
strides
2
max_pool_0/MaxPool∞
conv2D_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
conv2D_1/Conv2D/ReadVariableOp‘
conv2D_1/Conv2DConv2Dmax_pool_0/MaxPool:output:0&conv2D_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@*
paddingVALID*
strides
2
conv2D_1/Conv2DІ
conv2D_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2D_1/BiasAdd/ReadVariableOpђ
conv2D_1/BiasAddBiasAddconv2D_1/Conv2D:output:0'conv2D_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@2
conv2D_1/BiasAddГ
bn_1/ReadVariableOpReadVariableOpbn_1_readvariableop_resource*
_output_shapes
:@*
dtype02
bn_1/ReadVariableOpЙ
bn_1/ReadVariableOp_1ReadVariableOpbn_1_readvariableop_1_resource*
_output_shapes
:@*
dtype02
bn_1/ReadVariableOp_1ґ
$bn_1/FusedBatchNormV3/ReadVariableOpReadVariableOp-bn_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02&
$bn_1/FusedBatchNormV3/ReadVariableOpЉ
&bn_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp/bn_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&bn_1/FusedBatchNormV3/ReadVariableOp_1ы
bn_1/FusedBatchNormV3FusedBatchNormV3conv2D_1/BiasAdd:output:0bn_1/ReadVariableOp:value:0bn_1/ReadVariableOp_1:value:0,bn_1/FusedBatchNormV3/ReadVariableOp:value:0.bn_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
bn_1/FusedBatchNormV3Г
activation_1/ReluRelubn_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€@2
activation_1/ReluЅ
max_pool_1/MaxPoolMaxPoolactivation_1/Relu:activations:0*/
_output_shapes
:€€€€€€€€€@*
ksize
*
paddingVALID*
strides
2
max_pool_1/MaxPool±
conv2D_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype02 
conv2D_2/Conv2D/ReadVariableOp’
conv2D_2/Conv2DConv2Dmax_pool_1/MaxPool:output:0&conv2D_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€

А*
paddingVALID*
strides
2
conv2D_2/Conv2D®
conv2D_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
conv2D_2/BiasAdd/ReadVariableOp≠
conv2D_2/BiasAddBiasAddconv2D_2/Conv2D:output:0'conv2D_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€

А2
conv2D_2/BiasAddД
bn_2/ReadVariableOpReadVariableOpbn_2_readvariableop_resource*
_output_shapes	
:А*
dtype02
bn_2/ReadVariableOpК
bn_2/ReadVariableOp_1ReadVariableOpbn_2_readvariableop_1_resource*
_output_shapes	
:А*
dtype02
bn_2/ReadVariableOp_1Ј
$bn_2/FusedBatchNormV3/ReadVariableOpReadVariableOp-bn_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02&
$bn_2/FusedBatchNormV3/ReadVariableOpљ
&bn_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp/bn_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02(
&bn_2/FusedBatchNormV3/ReadVariableOp_1А
bn_2/FusedBatchNormV3FusedBatchNormV3conv2D_2/BiasAdd:output:0bn_2/ReadVariableOp:value:0bn_2/ReadVariableOp_1:value:0,bn_2/FusedBatchNormV3/ReadVariableOp:value:0.bn_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:€€€€€€€€€

А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
bn_2/FusedBatchNormV3Д
activation_2/ReluRelubn_2/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:€€€€€€€€€

А2
activation_2/Relu¬
max_pool_2/MaxPoolMaxPoolactivation_2/Relu:activations:0*0
_output_shapes
:€€€€€€€€€А*
ksize
*
paddingVALID*
strides
2
max_pool_2/MaxPoolo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€А  2
flatten/ConstХ
flatten/ReshapeReshapemax_pool_2/MaxPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
flatten/ReshapeЧ
fc/MatMul/ReadVariableOpReadVariableOp!fc_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02
fc/MatMul/ReadVariableOpО
	fc/MatMulMatMulflatten/Reshape:output:0 fc/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
	fc/MatMulХ
fc/BiasAdd/ReadVariableOpReadVariableOp"fc_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
fc/BiasAdd/ReadVariableOpН

fc/BiasAddBiasAddfc/MatMul:product:0!fc/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2

fc/BiasAddj

fc/SoftmaxSoftmaxfc/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2

fc/Softmaxh
IdentityIdentityfc/Softmax:softmax:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:€€€€€€€€€@@:::::::::::::::::::::W S
/
_output_shapes
:€€€€€€€€€@@
 
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
: 
№
}
(__inference_conv2D_0_layer_call_fn_10976

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallл
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_conv2D_0_layer_call_and_return_conditional_losses_109662
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
З
J
.__inference_zero_padding2d_layer_call_fn_10955

inputs
identity»
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_zero_padding2d_layer_call_and_return_conditional_losses_109492
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ґ
ш
?__inference_bn_0_layer_call_and_return_conditional_losses_12574

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1 
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€@@ : : : : :*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€@@ 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€@@ :::::W S
/
_output_shapes
:€€€€€€€€€@@ 
 
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
г
e
I__inference_zero_padding2d_layer_call_and_return_conditional_losses_10949

inputs
identityН
Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             2
Pad/paddingsЕ
PadPadinputsPad/paddings:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2
PadГ
IdentityIdentityPad:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Л
ш
?__inference_bn_2_layer_call_and_return_conditional_losses_12843

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1®
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpЃ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1б
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3Г
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:::::j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
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
к
w
"__inference_fc_layer_call_fn_12985

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallЋ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*F
fAR?
=__inference_fc_layer_call_and_return_conditional_losses_117712
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€А::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ў
c
G__inference_activation_2_layer_call_and_return_conditional_losses_12949

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:€€€€€€€€€

А2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:€€€€€€€€€

А2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€

А:X T
0
_output_shapes
:€€€€€€€€€

А
 
_user_specified_nameinputs
”
a
E__inference_activation_layer_call_and_return_conditional_losses_12605

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:€€€€€€€€€@@ 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€@@ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€@@ :W S
/
_output_shapes
:€€€€€€€€€@@ 
 
_user_specified_nameinputs
Ц
H
,__inference_activation_1_layer_call_fn_12782

inputs
identityЂ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_116342
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€@:W S
/
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
ќ
Р
#__inference_signature_wrapper_12147
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

unknown_18
identityИҐStatefulPartitionedCallҐ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€*6
_read_only_resource_inputs
	
**
config_proto

GPU 

CPU2J 8*)
f$R"
 __inference__wrapped_model_109422
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:€€€€€€€€€@@::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:€€€€€€€€€@@
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
: 
™Щ
„	
F__inference_CatDogModel_layer_call_and_return_conditional_losses_12267

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
/bn_2_fusedbatchnormv3_readvariableop_1_resource%
!fc_matmul_readvariableop_resource&
"fc_biasadd_readvariableop_resource
identityИҐ(bn_0/AssignMovingAvg/AssignSubVariableOpҐ*bn_0/AssignMovingAvg_1/AssignSubVariableOpҐ(bn_1/AssignMovingAvg/AssignSubVariableOpҐ*bn_1/AssignMovingAvg_1/AssignSubVariableOpҐ(bn_2/AssignMovingAvg/AssignSubVariableOpҐ*bn_2/AssignMovingAvg_1/AssignSubVariableOpЂ
zero_padding2d/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             2
zero_padding2d/Pad/paddingsЧ
zero_padding2d/PadPadinputs$zero_padding2d/Pad/paddings:output:0*
T0*/
_output_shapes
:€€€€€€€€€DD2
zero_padding2d/Pad∞
conv2D_0/Conv2D/ReadVariableOpReadVariableOp'conv2d_0_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2D_0/Conv2D/ReadVariableOp‘
conv2D_0/Conv2DConv2Dzero_padding2d/Pad:output:0&conv2D_0/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@@ *
paddingVALID*
strides
2
conv2D_0/Conv2DІ
conv2D_0/BiasAdd/ReadVariableOpReadVariableOp(conv2d_0_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2D_0/BiasAdd/ReadVariableOpђ
conv2D_0/BiasAddBiasAddconv2D_0/Conv2D:output:0'conv2D_0/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@@ 2
conv2D_0/BiasAddГ
bn_0/ReadVariableOpReadVariableOpbn_0_readvariableop_resource*
_output_shapes
: *
dtype02
bn_0/ReadVariableOpЙ
bn_0/ReadVariableOp_1ReadVariableOpbn_0_readvariableop_1_resource*
_output_shapes
: *
dtype02
bn_0/ReadVariableOp_1ґ
$bn_0/FusedBatchNormV3/ReadVariableOpReadVariableOp-bn_0_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02&
$bn_0/FusedBatchNormV3/ReadVariableOpЉ
&bn_0/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp/bn_0_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&bn_0/FusedBatchNormV3/ReadVariableOp_1и
bn_0/FusedBatchNormV3FusedBatchNormV3conv2D_0/BiasAdd:output:0bn_0/ReadVariableOp:value:0bn_0/ReadVariableOp_1:value:0,bn_0/FusedBatchNormV3/ReadVariableOp:value:0.bn_0/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€@@ : : : : :*
epsilon%oГ:2
bn_0/FusedBatchNormV3]

bn_0/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2

bn_0/Constњ
bn_0/AssignMovingAvg/sub/xConst*@
_class6
42loc:@bn_0/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
bn_0/AssignMovingAvg/sub/xЎ
bn_0/AssignMovingAvg/subSub#bn_0/AssignMovingAvg/sub/x:output:0bn_0/Const:output:0*
T0*@
_class6
42loc:@bn_0/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
bn_0/AssignMovingAvg/subі
#bn_0/AssignMovingAvg/ReadVariableOpReadVariableOp-bn_0_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02%
#bn_0/AssignMovingAvg/ReadVariableOpч
bn_0/AssignMovingAvg/sub_1Sub+bn_0/AssignMovingAvg/ReadVariableOp:value:0"bn_0/FusedBatchNormV3:batch_mean:0*
T0*@
_class6
42loc:@bn_0/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
bn_0/AssignMovingAvg/sub_1а
bn_0/AssignMovingAvg/mulMulbn_0/AssignMovingAvg/sub_1:z:0bn_0/AssignMovingAvg/sub:z:0*
T0*@
_class6
42loc:@bn_0/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
bn_0/AssignMovingAvg/mulк
(bn_0/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp-bn_0_fusedbatchnormv3_readvariableop_resourcebn_0/AssignMovingAvg/mul:z:0$^bn_0/AssignMovingAvg/ReadVariableOp%^bn_0/FusedBatchNormV3/ReadVariableOp*@
_class6
42loc:@bn_0/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02*
(bn_0/AssignMovingAvg/AssignSubVariableOp≈
bn_0/AssignMovingAvg_1/sub/xConst*B
_class8
64loc:@bn_0/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
bn_0/AssignMovingAvg_1/sub/xа
bn_0/AssignMovingAvg_1/subSub%bn_0/AssignMovingAvg_1/sub/x:output:0bn_0/Const:output:0*
T0*B
_class8
64loc:@bn_0/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
bn_0/AssignMovingAvg_1/subЇ
%bn_0/AssignMovingAvg_1/ReadVariableOpReadVariableOp/bn_0_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02'
%bn_0/AssignMovingAvg_1/ReadVariableOpГ
bn_0/AssignMovingAvg_1/sub_1Sub-bn_0/AssignMovingAvg_1/ReadVariableOp:value:0&bn_0/FusedBatchNormV3:batch_variance:0*
T0*B
_class8
64loc:@bn_0/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
bn_0/AssignMovingAvg_1/sub_1к
bn_0/AssignMovingAvg_1/mulMul bn_0/AssignMovingAvg_1/sub_1:z:0bn_0/AssignMovingAvg_1/sub:z:0*
T0*B
_class8
64loc:@bn_0/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
bn_0/AssignMovingAvg_1/mulш
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
:€€€€€€€€€@@ 2
activation/Reluњ
max_pool_0/MaxPoolMaxPoolactivation/Relu:activations:0*/
_output_shapes
:€€€€€€€€€   *
ksize
*
paddingVALID*
strides
2
max_pool_0/MaxPool∞
conv2D_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
conv2D_1/Conv2D/ReadVariableOp‘
conv2D_1/Conv2DConv2Dmax_pool_0/MaxPool:output:0&conv2D_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@*
paddingVALID*
strides
2
conv2D_1/Conv2DІ
conv2D_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2D_1/BiasAdd/ReadVariableOpђ
conv2D_1/BiasAddBiasAddconv2D_1/Conv2D:output:0'conv2D_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@2
conv2D_1/BiasAddГ
bn_1/ReadVariableOpReadVariableOpbn_1_readvariableop_resource*
_output_shapes
:@*
dtype02
bn_1/ReadVariableOpЙ
bn_1/ReadVariableOp_1ReadVariableOpbn_1_readvariableop_1_resource*
_output_shapes
:@*
dtype02
bn_1/ReadVariableOp_1ґ
$bn_1/FusedBatchNormV3/ReadVariableOpReadVariableOp-bn_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02&
$bn_1/FusedBatchNormV3/ReadVariableOpЉ
&bn_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp/bn_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&bn_1/FusedBatchNormV3/ReadVariableOp_1и
bn_1/FusedBatchNormV3FusedBatchNormV3conv2D_1/BiasAdd:output:0bn_1/ReadVariableOp:value:0bn_1/ReadVariableOp_1:value:0,bn_1/FusedBatchNormV3/ReadVariableOp:value:0.bn_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:2
bn_1/FusedBatchNormV3]

bn_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2

bn_1/Constњ
bn_1/AssignMovingAvg/sub/xConst*@
_class6
42loc:@bn_1/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
bn_1/AssignMovingAvg/sub/xЎ
bn_1/AssignMovingAvg/subSub#bn_1/AssignMovingAvg/sub/x:output:0bn_1/Const:output:0*
T0*@
_class6
42loc:@bn_1/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
bn_1/AssignMovingAvg/subі
#bn_1/AssignMovingAvg/ReadVariableOpReadVariableOp-bn_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02%
#bn_1/AssignMovingAvg/ReadVariableOpч
bn_1/AssignMovingAvg/sub_1Sub+bn_1/AssignMovingAvg/ReadVariableOp:value:0"bn_1/FusedBatchNormV3:batch_mean:0*
T0*@
_class6
42loc:@bn_1/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
bn_1/AssignMovingAvg/sub_1а
bn_1/AssignMovingAvg/mulMulbn_1/AssignMovingAvg/sub_1:z:0bn_1/AssignMovingAvg/sub:z:0*
T0*@
_class6
42loc:@bn_1/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
bn_1/AssignMovingAvg/mulк
(bn_1/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp-bn_1_fusedbatchnormv3_readvariableop_resourcebn_1/AssignMovingAvg/mul:z:0$^bn_1/AssignMovingAvg/ReadVariableOp%^bn_1/FusedBatchNormV3/ReadVariableOp*@
_class6
42loc:@bn_1/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02*
(bn_1/AssignMovingAvg/AssignSubVariableOp≈
bn_1/AssignMovingAvg_1/sub/xConst*B
_class8
64loc:@bn_1/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
bn_1/AssignMovingAvg_1/sub/xа
bn_1/AssignMovingAvg_1/subSub%bn_1/AssignMovingAvg_1/sub/x:output:0bn_1/Const:output:0*
T0*B
_class8
64loc:@bn_1/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
bn_1/AssignMovingAvg_1/subЇ
%bn_1/AssignMovingAvg_1/ReadVariableOpReadVariableOp/bn_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02'
%bn_1/AssignMovingAvg_1/ReadVariableOpГ
bn_1/AssignMovingAvg_1/sub_1Sub-bn_1/AssignMovingAvg_1/ReadVariableOp:value:0&bn_1/FusedBatchNormV3:batch_variance:0*
T0*B
_class8
64loc:@bn_1/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
bn_1/AssignMovingAvg_1/sub_1к
bn_1/AssignMovingAvg_1/mulMul bn_1/AssignMovingAvg_1/sub_1:z:0bn_1/AssignMovingAvg_1/sub:z:0*
T0*B
_class8
64loc:@bn_1/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
bn_1/AssignMovingAvg_1/mulш
*bn_1/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp/bn_1_fusedbatchnormv3_readvariableop_1_resourcebn_1/AssignMovingAvg_1/mul:z:0&^bn_1/AssignMovingAvg_1/ReadVariableOp'^bn_1/FusedBatchNormV3/ReadVariableOp_1*B
_class8
64loc:@bn_1/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02,
*bn_1/AssignMovingAvg_1/AssignSubVariableOpГ
activation_1/ReluRelubn_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€@2
activation_1/ReluЅ
max_pool_1/MaxPoolMaxPoolactivation_1/Relu:activations:0*/
_output_shapes
:€€€€€€€€€@*
ksize
*
paddingVALID*
strides
2
max_pool_1/MaxPool±
conv2D_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype02 
conv2D_2/Conv2D/ReadVariableOp’
conv2D_2/Conv2DConv2Dmax_pool_1/MaxPool:output:0&conv2D_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€

А*
paddingVALID*
strides
2
conv2D_2/Conv2D®
conv2D_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
conv2D_2/BiasAdd/ReadVariableOp≠
conv2D_2/BiasAddBiasAddconv2D_2/Conv2D:output:0'conv2D_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€

А2
conv2D_2/BiasAddД
bn_2/ReadVariableOpReadVariableOpbn_2_readvariableop_resource*
_output_shapes	
:А*
dtype02
bn_2/ReadVariableOpК
bn_2/ReadVariableOp_1ReadVariableOpbn_2_readvariableop_1_resource*
_output_shapes	
:А*
dtype02
bn_2/ReadVariableOp_1Ј
$bn_2/FusedBatchNormV3/ReadVariableOpReadVariableOp-bn_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02&
$bn_2/FusedBatchNormV3/ReadVariableOpљ
&bn_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp/bn_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02(
&bn_2/FusedBatchNormV3/ReadVariableOp_1н
bn_2/FusedBatchNormV3FusedBatchNormV3conv2D_2/BiasAdd:output:0bn_2/ReadVariableOp:value:0bn_2/ReadVariableOp_1:value:0,bn_2/FusedBatchNormV3/ReadVariableOp:value:0.bn_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:€€€€€€€€€

А:А:А:А:А:*
epsilon%oГ:2
bn_2/FusedBatchNormV3]

bn_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2

bn_2/Constњ
bn_2/AssignMovingAvg/sub/xConst*@
_class6
42loc:@bn_2/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
bn_2/AssignMovingAvg/sub/xЎ
bn_2/AssignMovingAvg/subSub#bn_2/AssignMovingAvg/sub/x:output:0bn_2/Const:output:0*
T0*@
_class6
42loc:@bn_2/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
bn_2/AssignMovingAvg/subµ
#bn_2/AssignMovingAvg/ReadVariableOpReadVariableOp-bn_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02%
#bn_2/AssignMovingAvg/ReadVariableOpш
bn_2/AssignMovingAvg/sub_1Sub+bn_2/AssignMovingAvg/ReadVariableOp:value:0"bn_2/FusedBatchNormV3:batch_mean:0*
T0*@
_class6
42loc:@bn_2/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:А2
bn_2/AssignMovingAvg/sub_1б
bn_2/AssignMovingAvg/mulMulbn_2/AssignMovingAvg/sub_1:z:0bn_2/AssignMovingAvg/sub:z:0*
T0*@
_class6
42loc:@bn_2/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:А2
bn_2/AssignMovingAvg/mulк
(bn_2/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp-bn_2_fusedbatchnormv3_readvariableop_resourcebn_2/AssignMovingAvg/mul:z:0$^bn_2/AssignMovingAvg/ReadVariableOp%^bn_2/FusedBatchNormV3/ReadVariableOp*@
_class6
42loc:@bn_2/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02*
(bn_2/AssignMovingAvg/AssignSubVariableOp≈
bn_2/AssignMovingAvg_1/sub/xConst*B
_class8
64loc:@bn_2/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
bn_2/AssignMovingAvg_1/sub/xа
bn_2/AssignMovingAvg_1/subSub%bn_2/AssignMovingAvg_1/sub/x:output:0bn_2/Const:output:0*
T0*B
_class8
64loc:@bn_2/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
bn_2/AssignMovingAvg_1/subї
%bn_2/AssignMovingAvg_1/ReadVariableOpReadVariableOp/bn_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02'
%bn_2/AssignMovingAvg_1/ReadVariableOpД
bn_2/AssignMovingAvg_1/sub_1Sub-bn_2/AssignMovingAvg_1/ReadVariableOp:value:0&bn_2/FusedBatchNormV3:batch_variance:0*
T0*B
_class8
64loc:@bn_2/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:А2
bn_2/AssignMovingAvg_1/sub_1л
bn_2/AssignMovingAvg_1/mulMul bn_2/AssignMovingAvg_1/sub_1:z:0bn_2/AssignMovingAvg_1/sub:z:0*
T0*B
_class8
64loc:@bn_2/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:А2
bn_2/AssignMovingAvg_1/mulш
*bn_2/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp/bn_2_fusedbatchnormv3_readvariableop_1_resourcebn_2/AssignMovingAvg_1/mul:z:0&^bn_2/AssignMovingAvg_1/ReadVariableOp'^bn_2/FusedBatchNormV3/ReadVariableOp_1*B
_class8
64loc:@bn_2/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02,
*bn_2/AssignMovingAvg_1/AssignSubVariableOpД
activation_2/ReluRelubn_2/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:€€€€€€€€€

А2
activation_2/Relu¬
max_pool_2/MaxPoolMaxPoolactivation_2/Relu:activations:0*0
_output_shapes
:€€€€€€€€€А*
ksize
*
paddingVALID*
strides
2
max_pool_2/MaxPoolo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€А  2
flatten/ConstХ
flatten/ReshapeReshapemax_pool_2/MaxPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
flatten/ReshapeЧ
fc/MatMul/ReadVariableOpReadVariableOp!fc_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02
fc/MatMul/ReadVariableOpО
	fc/MatMulMatMulflatten/Reshape:output:0 fc/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
	fc/MatMulХ
fc/BiasAdd/ReadVariableOpReadVariableOp"fc_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
fc/BiasAdd/ReadVariableOpН

fc/BiasAddBiasAddfc/MatMul:product:0!fc/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2

fc/BiasAddj

fc/SoftmaxSoftmaxfc/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2

fc/Softmaxр
IdentityIdentityfc/Softmax:softmax:0)^bn_0/AssignMovingAvg/AssignSubVariableOp+^bn_0/AssignMovingAvg_1/AssignSubVariableOp)^bn_1/AssignMovingAvg/AssignSubVariableOp+^bn_1/AssignMovingAvg_1/AssignSubVariableOp)^bn_2/AssignMovingAvg/AssignSubVariableOp+^bn_2/AssignMovingAvg_1/AssignSubVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:€€€€€€€€€@@::::::::::::::::::::2T
(bn_0/AssignMovingAvg/AssignSubVariableOp(bn_0/AssignMovingAvg/AssignSubVariableOp2X
*bn_0/AssignMovingAvg_1/AssignSubVariableOp*bn_0/AssignMovingAvg_1/AssignSubVariableOp2T
(bn_1/AssignMovingAvg/AssignSubVariableOp(bn_1/AssignMovingAvg/AssignSubVariableOp2X
*bn_1/AssignMovingAvg_1/AssignSubVariableOp*bn_1/AssignMovingAvg_1/AssignSubVariableOp2T
(bn_2/AssignMovingAvg/AssignSubVariableOp(bn_2/AssignMovingAvg/AssignSubVariableOp2X
*bn_2/AssignMovingAvg_1/AssignSubVariableOp*bn_2/AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:€€€€€€€€€@@
 
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
: 
З
Ч
$__inference_bn_1_layer_call_fn_12759

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallн
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*H
fCRA
?__inference_bn_1_layer_call_and_return_conditional_losses_115752
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€@
 
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
’$
∆
?__inference_bn_0_layer_call_and_return_conditional_losses_12481

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1…
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ : : : : :*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
Const∞
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xњ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub•
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpё
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub_1«
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/mul«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpґ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x«
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЂ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpк
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub_1—
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/mul’
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp–
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
 
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
€
ш
?__inference_bn_1_layer_call_and_return_conditional_losses_11250

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:::::i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
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
Я$
∆
?__inference_bn_2_layer_call_and_return_conditional_losses_12900

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1®
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpЃ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Љ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:€€€€€€€€€

А:А:А:А:А:*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
Const∞
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xњ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub¶
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02 
AssignMovingAvg/ReadVariableOpя
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:А2
AssignMovingAvg/sub_1»
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:А2
AssignMovingAvg/mul«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpґ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x«
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subђ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02"
 AssignMovingAvg_1/ReadVariableOpл
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:А2
AssignMovingAvg_1/sub_1“
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:А2
AssignMovingAvg_1/mul’
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpњ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*0
_output_shapes
:€€€€€€€€€

А2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:€€€€€€€€€

А::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:X T
0
_output_shapes
:€€€€€€€€€

А
 
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
€
F
*__inference_max_pool_0_layer_call_fn_11114

inputs
identityƒ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_max_pool_0_layer_call_and_return_conditional_losses_111082
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ґ
ш
?__inference_bn_1_layer_call_and_return_conditional_losses_11593

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1 
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€@:::::W S
/
_output_shapes
:€€€€€€€€€@
 
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
Я$
∆
?__inference_bn_2_layer_call_and_return_conditional_losses_11678

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1®
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpЃ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Љ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:€€€€€€€€€

А:А:А:А:А:*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
Const∞
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xњ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub¶
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02 
AssignMovingAvg/ReadVariableOpя
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:А2
AssignMovingAvg/sub_1»
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:А2
AssignMovingAvg/mul«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpґ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x«
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subђ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02"
 AssignMovingAvg_1/ReadVariableOpл
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:А2
AssignMovingAvg_1/sub_1“
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:А2
AssignMovingAvg_1/mul’
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpњ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*0
_output_shapes
:€€€€€€€€€

А2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:€€€€€€€€€

А::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:X T
0
_output_shapes
:€€€€€€€€€

А
 
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
Й
Ч
$__inference_bn_0_layer_call_fn_12600

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallп
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€@@ *&
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*H
fCRA
?__inference_bn_0_layer_call_and_return_conditional_losses_114902
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€@@ 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€@@ ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€@@ 
 
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
¬
ш
?__inference_bn_2_layer_call_and_return_conditional_losses_12918

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1®
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpЃ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ѕ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:€€€€€€€€€

А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:€€€€€€€€€

А2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:€€€€€€€€€

А:::::X T
0
_output_shapes
:€€€€€€€€€

А
 
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
¬
ш
?__inference_bn_2_layer_call_and_return_conditional_losses_11696

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1®
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpЃ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ѕ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:€€€€€€€€€

А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:€€€€€€€€€

А2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:€€€€€€€€€

А:::::X T
0
_output_shapes
:€€€€€€€€€

А
 
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
ѕ
Ч
$__inference_bn_0_layer_call_fn_12512

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCall€
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*H
fCRA
?__inference_bn_0_layer_call_and_return_conditional_losses_110602
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
 
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
’$
∆
?__inference_bn_1_layer_call_and_return_conditional_losses_11219

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1…
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
Const∞
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xњ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub•
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOpё
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/sub_1«
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/mul«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpґ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x«
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЂ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOpк
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/sub_1—
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/mul’
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp–
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
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
’$
∆
?__inference_bn_0_layer_call_and_return_conditional_losses_11060

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1…
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ : : : : :*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
Const∞
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xњ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub•
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpё
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub_1«
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/mul«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpґ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x«
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЂ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpк
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub_1—
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/mul’
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp–
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
 
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
ѕ
Ч
$__inference_bn_1_layer_call_fn_12684

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCall€
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*H
fCRA
?__inference_bn_1_layer_call_and_return_conditional_losses_112192
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
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
¶>
Э
F__inference_CatDogModel_layer_call_and_return_conditional_losses_11788
input_1
conv2d_0_11437
conv2d_0_11439

bn_0_11517

bn_0_11519

bn_0_11521

bn_0_11523
conv2d_1_11540
conv2d_1_11542

bn_1_11620

bn_1_11622

bn_1_11624

bn_1_11626
conv2d_2_11643
conv2d_2_11645

bn_2_11723

bn_2_11725

bn_2_11727

bn_2_11729
fc_11782
fc_11784
identityИҐbn_0/StatefulPartitionedCallҐbn_1/StatefulPartitionedCallҐbn_2/StatefulPartitionedCallҐ conv2D_0/StatefulPartitionedCallҐ conv2D_1/StatefulPartitionedCallҐ conv2D_2/StatefulPartitionedCallҐfc/StatefulPartitionedCallћ
zero_padding2d/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€DD* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_zero_padding2d_layer_call_and_return_conditional_losses_109492 
zero_padding2d/PartitionedCallШ
 conv2D_0/StatefulPartitionedCallStatefulPartitionedCall'zero_padding2d/PartitionedCall:output:0conv2d_0_11437conv2d_0_11439*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€@@ *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_conv2D_0_layer_call_and_return_conditional_losses_109662"
 conv2D_0/StatefulPartitionedCall†
bn_0/StatefulPartitionedCallStatefulPartitionedCall)conv2D_0/StatefulPartitionedCall:output:0
bn_0_11517
bn_0_11519
bn_0_11521
bn_0_11523*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€@@ *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*H
fCRA
?__inference_bn_0_layer_call_and_return_conditional_losses_114722
bn_0/StatefulPartitionedCallё
activation/PartitionedCallPartitionedCall%bn_0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€@@ * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_115312
activation/PartitionedCall№
max_pool_0/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€   * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_max_pool_0_layer_call_and_return_conditional_losses_111082
max_pool_0/PartitionedCallФ
 conv2D_1/StatefulPartitionedCallStatefulPartitionedCall#max_pool_0/PartitionedCall:output:0conv2d_1_11540conv2d_1_11542*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_conv2D_1_layer_call_and_return_conditional_losses_111252"
 conv2D_1/StatefulPartitionedCall†
bn_1/StatefulPartitionedCallStatefulPartitionedCall)conv2D_1/StatefulPartitionedCall:output:0
bn_1_11620
bn_1_11622
bn_1_11624
bn_1_11626*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*H
fCRA
?__inference_bn_1_layer_call_and_return_conditional_losses_115752
bn_1/StatefulPartitionedCallд
activation_1/PartitionedCallPartitionedCall%bn_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_116342
activation_1/PartitionedCallё
max_pool_1/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_max_pool_1_layer_call_and_return_conditional_losses_112672
max_pool_1/PartitionedCallХ
 conv2D_2/StatefulPartitionedCallStatefulPartitionedCall#max_pool_1/PartitionedCall:output:0conv2d_2_11643conv2d_2_11645*
Tin
2*
Tout
2*0
_output_shapes
:€€€€€€€€€

А*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_conv2D_2_layer_call_and_return_conditional_losses_112842"
 conv2D_2/StatefulPartitionedCall°
bn_2/StatefulPartitionedCallStatefulPartitionedCall)conv2D_2/StatefulPartitionedCall:output:0
bn_2_11723
bn_2_11725
bn_2_11727
bn_2_11729*
Tin	
2*
Tout
2*0
_output_shapes
:€€€€€€€€€

А*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*H
fCRA
?__inference_bn_2_layer_call_and_return_conditional_losses_116782
bn_2/StatefulPartitionedCallе
activation_2/PartitionedCallPartitionedCall%bn_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*0
_output_shapes
:€€€€€€€€€

А* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*P
fKRI
G__inference_activation_2_layer_call_and_return_conditional_losses_117372
activation_2/PartitionedCallя
max_pool_2/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_max_pool_2_layer_call_and_return_conditional_losses_114262
max_pool_2/PartitionedCallћ
flatten/PartitionedCallPartitionedCall#max_pool_2/PartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_117522
flatten/PartitionedCallл
fc/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0fc_11782fc_11784*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*F
fAR?
=__inference_fc_layer_call_and_return_conditional_losses_117712
fc/StatefulPartitionedCallЏ
IdentityIdentity#fc/StatefulPartitionedCall:output:0^bn_0/StatefulPartitionedCall^bn_1/StatefulPartitionedCall^bn_2/StatefulPartitionedCall!^conv2D_0/StatefulPartitionedCall!^conv2D_1/StatefulPartitionedCall!^conv2D_2/StatefulPartitionedCall^fc/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:€€€€€€€€€@@::::::::::::::::::::2<
bn_0/StatefulPartitionedCallbn_0/StatefulPartitionedCall2<
bn_1/StatefulPartitionedCallbn_1/StatefulPartitionedCall2<
bn_2/StatefulPartitionedCallbn_2/StatefulPartitionedCall2D
 conv2D_0/StatefulPartitionedCall conv2D_0/StatefulPartitionedCall2D
 conv2D_1/StatefulPartitionedCall conv2D_1/StatefulPartitionedCall2D
 conv2D_2/StatefulPartitionedCall conv2D_2/StatefulPartitionedCall28
fc/StatefulPartitionedCallfc/StatefulPartitionedCall:X T
/
_output_shapes
:€€€€€€€€€@@
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
: 
€
ш
?__inference_bn_0_layer_call_and_return_conditional_losses_11091

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ : : : : :*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ :::::i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
 
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
£>
Ь
F__inference_CatDogModel_layer_call_and_return_conditional_losses_11909

inputs
conv2d_0_11854
conv2d_0_11856

bn_0_11859

bn_0_11861

bn_0_11863

bn_0_11865
conv2d_1_11870
conv2d_1_11872

bn_1_11875

bn_1_11877

bn_1_11879

bn_1_11881
conv2d_2_11886
conv2d_2_11888

bn_2_11891

bn_2_11893

bn_2_11895

bn_2_11897
fc_11903
fc_11905
identityИҐbn_0/StatefulPartitionedCallҐbn_1/StatefulPartitionedCallҐbn_2/StatefulPartitionedCallҐ conv2D_0/StatefulPartitionedCallҐ conv2D_1/StatefulPartitionedCallҐ conv2D_2/StatefulPartitionedCallҐfc/StatefulPartitionedCallЋ
zero_padding2d/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€DD* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_zero_padding2d_layer_call_and_return_conditional_losses_109492 
zero_padding2d/PartitionedCallШ
 conv2D_0/StatefulPartitionedCallStatefulPartitionedCall'zero_padding2d/PartitionedCall:output:0conv2d_0_11854conv2d_0_11856*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€@@ *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_conv2D_0_layer_call_and_return_conditional_losses_109662"
 conv2D_0/StatefulPartitionedCall†
bn_0/StatefulPartitionedCallStatefulPartitionedCall)conv2D_0/StatefulPartitionedCall:output:0
bn_0_11859
bn_0_11861
bn_0_11863
bn_0_11865*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€@@ *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*H
fCRA
?__inference_bn_0_layer_call_and_return_conditional_losses_114722
bn_0/StatefulPartitionedCallё
activation/PartitionedCallPartitionedCall%bn_0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€@@ * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_115312
activation/PartitionedCall№
max_pool_0/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€   * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_max_pool_0_layer_call_and_return_conditional_losses_111082
max_pool_0/PartitionedCallФ
 conv2D_1/StatefulPartitionedCallStatefulPartitionedCall#max_pool_0/PartitionedCall:output:0conv2d_1_11870conv2d_1_11872*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_conv2D_1_layer_call_and_return_conditional_losses_111252"
 conv2D_1/StatefulPartitionedCall†
bn_1/StatefulPartitionedCallStatefulPartitionedCall)conv2D_1/StatefulPartitionedCall:output:0
bn_1_11875
bn_1_11877
bn_1_11879
bn_1_11881*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*H
fCRA
?__inference_bn_1_layer_call_and_return_conditional_losses_115752
bn_1/StatefulPartitionedCallд
activation_1/PartitionedCallPartitionedCall%bn_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_116342
activation_1/PartitionedCallё
max_pool_1/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_max_pool_1_layer_call_and_return_conditional_losses_112672
max_pool_1/PartitionedCallХ
 conv2D_2/StatefulPartitionedCallStatefulPartitionedCall#max_pool_1/PartitionedCall:output:0conv2d_2_11886conv2d_2_11888*
Tin
2*
Tout
2*0
_output_shapes
:€€€€€€€€€

А*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_conv2D_2_layer_call_and_return_conditional_losses_112842"
 conv2D_2/StatefulPartitionedCall°
bn_2/StatefulPartitionedCallStatefulPartitionedCall)conv2D_2/StatefulPartitionedCall:output:0
bn_2_11891
bn_2_11893
bn_2_11895
bn_2_11897*
Tin	
2*
Tout
2*0
_output_shapes
:€€€€€€€€€

А*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*H
fCRA
?__inference_bn_2_layer_call_and_return_conditional_losses_116782
bn_2/StatefulPartitionedCallе
activation_2/PartitionedCallPartitionedCall%bn_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*0
_output_shapes
:€€€€€€€€€

А* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*P
fKRI
G__inference_activation_2_layer_call_and_return_conditional_losses_117372
activation_2/PartitionedCallя
max_pool_2/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_max_pool_2_layer_call_and_return_conditional_losses_114262
max_pool_2/PartitionedCallћ
flatten/PartitionedCallPartitionedCall#max_pool_2/PartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_117522
flatten/PartitionedCallл
fc/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0fc_11903fc_11905*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*F
fAR?
=__inference_fc_layer_call_and_return_conditional_losses_117712
fc/StatefulPartitionedCallЏ
IdentityIdentity#fc/StatefulPartitionedCall:output:0^bn_0/StatefulPartitionedCall^bn_1/StatefulPartitionedCall^bn_2/StatefulPartitionedCall!^conv2D_0/StatefulPartitionedCall!^conv2D_1/StatefulPartitionedCall!^conv2D_2/StatefulPartitionedCall^fc/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:€€€€€€€€€@@::::::::::::::::::::2<
bn_0/StatefulPartitionedCallbn_0/StatefulPartitionedCall2<
bn_1/StatefulPartitionedCallbn_1/StatefulPartitionedCall2<
bn_2/StatefulPartitionedCallbn_2/StatefulPartitionedCall2D
 conv2D_0/StatefulPartitionedCall conv2D_0/StatefulPartitionedCall2D
 conv2D_1/StatefulPartitionedCall conv2D_1/StatefulPartitionedCall2D
 conv2D_2/StatefulPartitionedCall conv2D_2/StatefulPartitionedCall28
fc/StatefulPartitionedCallfc/StatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€@@
 
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
: 
Н
Ч
$__inference_bn_2_layer_call_fn_12944

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallр
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*0
_output_shapes
:€€€€€€€€€

А*&
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*H
fCRA
?__inference_bn_2_layer_call_and_return_conditional_losses_116962
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:€€€€€€€€€

А2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:€€€€€€€€€

А::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:€€€€€€€€€

А
 
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
й
•
=__inference_fc_layer_call_and_return_conditional_losses_12976

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€А:::P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
€
ш
?__inference_bn_1_layer_call_and_return_conditional_losses_12671

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:::::i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
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
Љ
^
B__inference_flatten_layer_call_and_return_conditional_losses_11752

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€А  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€А:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
©>
Ь
F__inference_CatDogModel_layer_call_and_return_conditional_losses_12013

inputs
conv2d_0_11958
conv2d_0_11960

bn_0_11963

bn_0_11965

bn_0_11967

bn_0_11969
conv2d_1_11974
conv2d_1_11976

bn_1_11979

bn_1_11981

bn_1_11983

bn_1_11985
conv2d_2_11990
conv2d_2_11992

bn_2_11995

bn_2_11997

bn_2_11999

bn_2_12001
fc_12007
fc_12009
identityИҐbn_0/StatefulPartitionedCallҐbn_1/StatefulPartitionedCallҐbn_2/StatefulPartitionedCallҐ conv2D_0/StatefulPartitionedCallҐ conv2D_1/StatefulPartitionedCallҐ conv2D_2/StatefulPartitionedCallҐfc/StatefulPartitionedCallЋ
zero_padding2d/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€DD* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_zero_padding2d_layer_call_and_return_conditional_losses_109492 
zero_padding2d/PartitionedCallШ
 conv2D_0/StatefulPartitionedCallStatefulPartitionedCall'zero_padding2d/PartitionedCall:output:0conv2d_0_11958conv2d_0_11960*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€@@ *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_conv2D_0_layer_call_and_return_conditional_losses_109662"
 conv2D_0/StatefulPartitionedCallҐ
bn_0/StatefulPartitionedCallStatefulPartitionedCall)conv2D_0/StatefulPartitionedCall:output:0
bn_0_11963
bn_0_11965
bn_0_11967
bn_0_11969*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€@@ *&
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*H
fCRA
?__inference_bn_0_layer_call_and_return_conditional_losses_114902
bn_0/StatefulPartitionedCallё
activation/PartitionedCallPartitionedCall%bn_0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€@@ * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_115312
activation/PartitionedCall№
max_pool_0/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€   * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_max_pool_0_layer_call_and_return_conditional_losses_111082
max_pool_0/PartitionedCallФ
 conv2D_1/StatefulPartitionedCallStatefulPartitionedCall#max_pool_0/PartitionedCall:output:0conv2d_1_11974conv2d_1_11976*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_conv2D_1_layer_call_and_return_conditional_losses_111252"
 conv2D_1/StatefulPartitionedCallҐ
bn_1/StatefulPartitionedCallStatefulPartitionedCall)conv2D_1/StatefulPartitionedCall:output:0
bn_1_11979
bn_1_11981
bn_1_11983
bn_1_11985*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€@*&
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*H
fCRA
?__inference_bn_1_layer_call_and_return_conditional_losses_115932
bn_1/StatefulPartitionedCallд
activation_1/PartitionedCallPartitionedCall%bn_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_116342
activation_1/PartitionedCallё
max_pool_1/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_max_pool_1_layer_call_and_return_conditional_losses_112672
max_pool_1/PartitionedCallХ
 conv2D_2/StatefulPartitionedCallStatefulPartitionedCall#max_pool_1/PartitionedCall:output:0conv2d_2_11990conv2d_2_11992*
Tin
2*
Tout
2*0
_output_shapes
:€€€€€€€€€

А*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_conv2D_2_layer_call_and_return_conditional_losses_112842"
 conv2D_2/StatefulPartitionedCall£
bn_2/StatefulPartitionedCallStatefulPartitionedCall)conv2D_2/StatefulPartitionedCall:output:0
bn_2_11995
bn_2_11997
bn_2_11999
bn_2_12001*
Tin	
2*
Tout
2*0
_output_shapes
:€€€€€€€€€

А*&
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*H
fCRA
?__inference_bn_2_layer_call_and_return_conditional_losses_116962
bn_2/StatefulPartitionedCallе
activation_2/PartitionedCallPartitionedCall%bn_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*0
_output_shapes
:€€€€€€€€€

А* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*P
fKRI
G__inference_activation_2_layer_call_and_return_conditional_losses_117372
activation_2/PartitionedCallя
max_pool_2/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_max_pool_2_layer_call_and_return_conditional_losses_114262
max_pool_2/PartitionedCallћ
flatten/PartitionedCallPartitionedCall#max_pool_2/PartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_117522
flatten/PartitionedCallл
fc/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0fc_12007fc_12009*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*F
fAR?
=__inference_fc_layer_call_and_return_conditional_losses_117712
fc/StatefulPartitionedCallЏ
IdentityIdentity#fc/StatefulPartitionedCall:output:0^bn_0/StatefulPartitionedCall^bn_1/StatefulPartitionedCall^bn_2/StatefulPartitionedCall!^conv2D_0/StatefulPartitionedCall!^conv2D_1/StatefulPartitionedCall!^conv2D_2/StatefulPartitionedCall^fc/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:€€€€€€€€€@@::::::::::::::::::::2<
bn_0/StatefulPartitionedCallbn_0/StatefulPartitionedCall2<
bn_1/StatefulPartitionedCallbn_1/StatefulPartitionedCall2<
bn_2/StatefulPartitionedCallbn_2/StatefulPartitionedCall2D
 conv2D_0/StatefulPartitionedCall conv2D_0/StatefulPartitionedCall2D
 conv2D_1/StatefulPartitionedCall conv2D_1/StatefulPartitionedCall2D
 conv2D_2/StatefulPartitionedCall conv2D_2/StatefulPartitionedCall28
fc/StatefulPartitionedCallfc/StatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€@@
 
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
: 
”
Ч
$__inference_bn_2_layer_call_fn_12856

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallА
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*H
fCRA
?__inference_bn_2_layer_call_and_return_conditional_losses_113782
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
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
Н$
∆
?__inference_bn_0_layer_call_and_return_conditional_losses_11472

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ј
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€@@ : : : : :*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
Const∞
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xњ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub•
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpё
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub_1«
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/mul«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpґ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x«
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЂ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpк
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub_1—
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/mul’
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpЊ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*/
_output_shapes
:€€€€€€€€€@@ 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€@@ ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:€€€€€€€€€@@ 
 
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
й
•
=__inference_fc_layer_call_and_return_conditional_losses_11771

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€А:::P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Т
F
*__inference_activation_layer_call_fn_12610

inputs
identity©
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€@@ * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_115312
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:€€€€€€€€€@@ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€@@ :W S
/
_output_shapes
:€€€€€€€€€@@ 
 
_user_specified_nameinputs
€
F
*__inference_max_pool_1_layer_call_fn_11273

inputs
identityƒ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_max_pool_1_layer_call_and_return_conditional_losses_112672
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Л
Ч
$__inference_bn_2_layer_call_fn_12931

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallо
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*0
_output_shapes
:€€€€€€€€€

А*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*H
fCRA
?__inference_bn_2_layer_call_and_return_conditional_losses_116782
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:€€€€€€€€€

А2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:€€€€€€€€€

А::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:€€€€€€€€€

А
 
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
ы
a
E__inference_max_pool_1_layer_call_and_return_conditional_losses_11267

inputs
identity≠
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
№
}
(__inference_conv2D_1_layer_call_fn_11135

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallл
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_conv2D_1_layer_call_and_return_conditional_losses_111252
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
’
c
G__inference_activation_1_layer_call_and_return_conditional_losses_12777

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:€€€€€€€€€@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€@:W S
/
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
лY
Ъ	
 __inference__wrapped_model_10942
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
;catdogmodel_bn_2_fusedbatchnormv3_readvariableop_1_resource1
-catdogmodel_fc_matmul_readvariableop_resource2
.catdogmodel_fc_biasadd_readvariableop_resource
identityИ√
'CatDogModel/zero_padding2d/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             2)
'CatDogModel/zero_padding2d/Pad/paddingsЉ
CatDogModel/zero_padding2d/PadPadinput_10CatDogModel/zero_padding2d/Pad/paddings:output:0*
T0*/
_output_shapes
:€€€€€€€€€DD2 
CatDogModel/zero_padding2d/Pad‘
*CatDogModel/conv2D_0/Conv2D/ReadVariableOpReadVariableOp3catdogmodel_conv2d_0_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02,
*CatDogModel/conv2D_0/Conv2D/ReadVariableOpД
CatDogModel/conv2D_0/Conv2DConv2D'CatDogModel/zero_padding2d/Pad:output:02CatDogModel/conv2D_0/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@@ *
paddingVALID*
strides
2
CatDogModel/conv2D_0/Conv2DЋ
+CatDogModel/conv2D_0/BiasAdd/ReadVariableOpReadVariableOp4catdogmodel_conv2d_0_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+CatDogModel/conv2D_0/BiasAdd/ReadVariableOp№
CatDogModel/conv2D_0/BiasAddBiasAdd$CatDogModel/conv2D_0/Conv2D:output:03CatDogModel/conv2D_0/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@@ 2
CatDogModel/conv2D_0/BiasAddІ
CatDogModel/bn_0/ReadVariableOpReadVariableOp(catdogmodel_bn_0_readvariableop_resource*
_output_shapes
: *
dtype02!
CatDogModel/bn_0/ReadVariableOp≠
!CatDogModel/bn_0/ReadVariableOp_1ReadVariableOp*catdogmodel_bn_0_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!CatDogModel/bn_0/ReadVariableOp_1Џ
0CatDogModel/bn_0/FusedBatchNormV3/ReadVariableOpReadVariableOp9catdogmodel_bn_0_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype022
0CatDogModel/bn_0/FusedBatchNormV3/ReadVariableOpа
2CatDogModel/bn_0/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp;catdogmodel_bn_0_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype024
2CatDogModel/bn_0/FusedBatchNormV3/ReadVariableOp_1ѕ
!CatDogModel/bn_0/FusedBatchNormV3FusedBatchNormV3%CatDogModel/conv2D_0/BiasAdd:output:0'CatDogModel/bn_0/ReadVariableOp:value:0)CatDogModel/bn_0/ReadVariableOp_1:value:08CatDogModel/bn_0/FusedBatchNormV3/ReadVariableOp:value:0:CatDogModel/bn_0/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€@@ : : : : :*
epsilon%oГ:*
is_training( 2#
!CatDogModel/bn_0/FusedBatchNormV3£
CatDogModel/activation/ReluRelu%CatDogModel/bn_0/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€@@ 2
CatDogModel/activation/Reluг
CatDogModel/max_pool_0/MaxPoolMaxPool)CatDogModel/activation/Relu:activations:0*/
_output_shapes
:€€€€€€€€€   *
ksize
*
paddingVALID*
strides
2 
CatDogModel/max_pool_0/MaxPool‘
*CatDogModel/conv2D_1/Conv2D/ReadVariableOpReadVariableOp3catdogmodel_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02,
*CatDogModel/conv2D_1/Conv2D/ReadVariableOpД
CatDogModel/conv2D_1/Conv2DConv2D'CatDogModel/max_pool_0/MaxPool:output:02CatDogModel/conv2D_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@*
paddingVALID*
strides
2
CatDogModel/conv2D_1/Conv2DЋ
+CatDogModel/conv2D_1/BiasAdd/ReadVariableOpReadVariableOp4catdogmodel_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+CatDogModel/conv2D_1/BiasAdd/ReadVariableOp№
CatDogModel/conv2D_1/BiasAddBiasAdd$CatDogModel/conv2D_1/Conv2D:output:03CatDogModel/conv2D_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@2
CatDogModel/conv2D_1/BiasAddІ
CatDogModel/bn_1/ReadVariableOpReadVariableOp(catdogmodel_bn_1_readvariableop_resource*
_output_shapes
:@*
dtype02!
CatDogModel/bn_1/ReadVariableOp≠
!CatDogModel/bn_1/ReadVariableOp_1ReadVariableOp*catdogmodel_bn_1_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!CatDogModel/bn_1/ReadVariableOp_1Џ
0CatDogModel/bn_1/FusedBatchNormV3/ReadVariableOpReadVariableOp9catdogmodel_bn_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype022
0CatDogModel/bn_1/FusedBatchNormV3/ReadVariableOpа
2CatDogModel/bn_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp;catdogmodel_bn_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype024
2CatDogModel/bn_1/FusedBatchNormV3/ReadVariableOp_1ѕ
!CatDogModel/bn_1/FusedBatchNormV3FusedBatchNormV3%CatDogModel/conv2D_1/BiasAdd:output:0'CatDogModel/bn_1/ReadVariableOp:value:0)CatDogModel/bn_1/ReadVariableOp_1:value:08CatDogModel/bn_1/FusedBatchNormV3/ReadVariableOp:value:0:CatDogModel/bn_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( 2#
!CatDogModel/bn_1/FusedBatchNormV3І
CatDogModel/activation_1/ReluRelu%CatDogModel/bn_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€@2
CatDogModel/activation_1/Reluе
CatDogModel/max_pool_1/MaxPoolMaxPool+CatDogModel/activation_1/Relu:activations:0*/
_output_shapes
:€€€€€€€€€@*
ksize
*
paddingVALID*
strides
2 
CatDogModel/max_pool_1/MaxPool’
*CatDogModel/conv2D_2/Conv2D/ReadVariableOpReadVariableOp3catdogmodel_conv2d_2_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype02,
*CatDogModel/conv2D_2/Conv2D/ReadVariableOpЕ
CatDogModel/conv2D_2/Conv2DConv2D'CatDogModel/max_pool_1/MaxPool:output:02CatDogModel/conv2D_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€

А*
paddingVALID*
strides
2
CatDogModel/conv2D_2/Conv2Dћ
+CatDogModel/conv2D_2/BiasAdd/ReadVariableOpReadVariableOp4catdogmodel_conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02-
+CatDogModel/conv2D_2/BiasAdd/ReadVariableOpЁ
CatDogModel/conv2D_2/BiasAddBiasAdd$CatDogModel/conv2D_2/Conv2D:output:03CatDogModel/conv2D_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€

А2
CatDogModel/conv2D_2/BiasAdd®
CatDogModel/bn_2/ReadVariableOpReadVariableOp(catdogmodel_bn_2_readvariableop_resource*
_output_shapes	
:А*
dtype02!
CatDogModel/bn_2/ReadVariableOpЃ
!CatDogModel/bn_2/ReadVariableOp_1ReadVariableOp*catdogmodel_bn_2_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!CatDogModel/bn_2/ReadVariableOp_1џ
0CatDogModel/bn_2/FusedBatchNormV3/ReadVariableOpReadVariableOp9catdogmodel_bn_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype022
0CatDogModel/bn_2/FusedBatchNormV3/ReadVariableOpб
2CatDogModel/bn_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp;catdogmodel_bn_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype024
2CatDogModel/bn_2/FusedBatchNormV3/ReadVariableOp_1‘
!CatDogModel/bn_2/FusedBatchNormV3FusedBatchNormV3%CatDogModel/conv2D_2/BiasAdd:output:0'CatDogModel/bn_2/ReadVariableOp:value:0)CatDogModel/bn_2/ReadVariableOp_1:value:08CatDogModel/bn_2/FusedBatchNormV3/ReadVariableOp:value:0:CatDogModel/bn_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:€€€€€€€€€

А:А:А:А:А:*
epsilon%oГ:*
is_training( 2#
!CatDogModel/bn_2/FusedBatchNormV3®
CatDogModel/activation_2/ReluRelu%CatDogModel/bn_2/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:€€€€€€€€€

А2
CatDogModel/activation_2/Reluж
CatDogModel/max_pool_2/MaxPoolMaxPool+CatDogModel/activation_2/Relu:activations:0*0
_output_shapes
:€€€€€€€€€А*
ksize
*
paddingVALID*
strides
2 
CatDogModel/max_pool_2/MaxPoolЗ
CatDogModel/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€А  2
CatDogModel/flatten/Const≈
CatDogModel/flatten/ReshapeReshape'CatDogModel/max_pool_2/MaxPool:output:0"CatDogModel/flatten/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
CatDogModel/flatten/Reshapeї
$CatDogModel/fc/MatMul/ReadVariableOpReadVariableOp-catdogmodel_fc_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02&
$CatDogModel/fc/MatMul/ReadVariableOpЊ
CatDogModel/fc/MatMulMatMul$CatDogModel/flatten/Reshape:output:0,CatDogModel/fc/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
CatDogModel/fc/MatMulє
%CatDogModel/fc/BiasAdd/ReadVariableOpReadVariableOp.catdogmodel_fc_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%CatDogModel/fc/BiasAdd/ReadVariableOpљ
CatDogModel/fc/BiasAddBiasAddCatDogModel/fc/MatMul:product:0-CatDogModel/fc/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
CatDogModel/fc/BiasAddО
CatDogModel/fc/SoftmaxSoftmaxCatDogModel/fc/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
CatDogModel/fc/Softmaxt
IdentityIdentity CatDogModel/fc/Softmax:softmax:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:€€€€€€€€€@@:::::::::::::::::::::X T
/
_output_shapes
:€€€€€€€€€@@
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
: 
—
Ч
$__inference_bn_1_layer_call_fn_12697

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallБ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*&
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*H
fCRA
?__inference_bn_1_layer_call_and_return_conditional_losses_112502
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
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
у
Ч
+__inference_CatDogModel_layer_call_fn_12393

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

unknown_18
identityИҐStatefulPartitionedCallЅ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€*0
_read_only_resource_inputs
	
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_CatDogModel_layer_call_and_return_conditional_losses_119092
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:€€€€€€€€€@@::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€@@
 
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
: 
ґ
ш
?__inference_bn_1_layer_call_and_return_conditional_losses_12746

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1 
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€@:::::W S
/
_output_shapes
:€€€€€€€€€@
 
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
ы
a
E__inference_max_pool_0_layer_call_and_return_conditional_losses_11108

inputs
identity≠
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ь
Ш
+__inference_CatDogModel_layer_call_fn_12056
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

unknown_18
identityИҐStatefulPartitionedCall»
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€*6
_read_only_resource_inputs
	
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_CatDogModel_layer_call_and_return_conditional_losses_120132
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:€€€€€€€€€@@::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:€€€€€€€€€@@
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
: 
€
ш
?__inference_bn_0_layer_call_and_return_conditional_losses_12499

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ : : : : :*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ :::::i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
 
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
ї	
Ђ
C__inference_conv2D_1_layer_call_and_return_conditional_losses_11125

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOpґ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
paddingVALID*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpЪ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2	
BiasAdd~
IdentityIdentityBiasAdd:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ :::i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
’
c
G__inference_activation_1_layer_call_and_return_conditional_losses_11634

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:€€€€€€€€€@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€@:W S
/
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
Н$
∆
?__inference_bn_1_layer_call_and_return_conditional_losses_12728

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ј
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
Const∞
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xњ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub•
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOpё
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/sub_1«
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/mul«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpґ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x«
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЂ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOpк
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/sub_1—
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/mul’
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpЊ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*/
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:€€€€€€€€€@
 
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
’$
∆
?__inference_bn_1_layer_call_and_return_conditional_losses_12653

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1…
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
Const∞
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xњ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub•
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOpё
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/sub_1«
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/mul«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpґ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x«
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЂ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOpк
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/sub_1—
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/mul’
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp–
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
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
—
Ч
$__inference_bn_0_layer_call_fn_12525

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallБ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ *&
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*H
fCRA
?__inference_bn_0_layer_call_and_return_conditional_losses_110912
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
 
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
ц
Ш
+__inference_CatDogModel_layer_call_fn_11952
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

unknown_18
identityИҐStatefulPartitionedCall¬
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€*0
_read_only_resource_inputs
	
**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_CatDogModel_layer_call_and_return_conditional_losses_119092
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:€€€€€€€€€@@::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:€€€€€€€€€@@
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
: 
ђ>
Э
F__inference_CatDogModel_layer_call_and_return_conditional_losses_11847
input_1
conv2d_0_11792
conv2d_0_11794

bn_0_11797

bn_0_11799

bn_0_11801

bn_0_11803
conv2d_1_11808
conv2d_1_11810

bn_1_11813

bn_1_11815

bn_1_11817

bn_1_11819
conv2d_2_11824
conv2d_2_11826

bn_2_11829

bn_2_11831

bn_2_11833

bn_2_11835
fc_11841
fc_11843
identityИҐbn_0/StatefulPartitionedCallҐbn_1/StatefulPartitionedCallҐbn_2/StatefulPartitionedCallҐ conv2D_0/StatefulPartitionedCallҐ conv2D_1/StatefulPartitionedCallҐ conv2D_2/StatefulPartitionedCallҐfc/StatefulPartitionedCallћ
zero_padding2d/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€DD* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*R
fMRK
I__inference_zero_padding2d_layer_call_and_return_conditional_losses_109492 
zero_padding2d/PartitionedCallШ
 conv2D_0/StatefulPartitionedCallStatefulPartitionedCall'zero_padding2d/PartitionedCall:output:0conv2d_0_11792conv2d_0_11794*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€@@ *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_conv2D_0_layer_call_and_return_conditional_losses_109662"
 conv2D_0/StatefulPartitionedCallҐ
bn_0/StatefulPartitionedCallStatefulPartitionedCall)conv2D_0/StatefulPartitionedCall:output:0
bn_0_11797
bn_0_11799
bn_0_11801
bn_0_11803*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€@@ *&
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*H
fCRA
?__inference_bn_0_layer_call_and_return_conditional_losses_114902
bn_0/StatefulPartitionedCallё
activation/PartitionedCallPartitionedCall%bn_0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€@@ * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_115312
activation/PartitionedCall№
max_pool_0/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€   * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_max_pool_0_layer_call_and_return_conditional_losses_111082
max_pool_0/PartitionedCallФ
 conv2D_1/StatefulPartitionedCallStatefulPartitionedCall#max_pool_0/PartitionedCall:output:0conv2d_1_11808conv2d_1_11810*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_conv2D_1_layer_call_and_return_conditional_losses_111252"
 conv2D_1/StatefulPartitionedCallҐ
bn_1/StatefulPartitionedCallStatefulPartitionedCall)conv2D_1/StatefulPartitionedCall:output:0
bn_1_11813
bn_1_11815
bn_1_11817
bn_1_11819*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€@*&
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*H
fCRA
?__inference_bn_1_layer_call_and_return_conditional_losses_115932
bn_1/StatefulPartitionedCallд
activation_1/PartitionedCallPartitionedCall%bn_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_116342
activation_1/PartitionedCallё
max_pool_1/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_max_pool_1_layer_call_and_return_conditional_losses_112672
max_pool_1/PartitionedCallХ
 conv2D_2/StatefulPartitionedCallStatefulPartitionedCall#max_pool_1/PartitionedCall:output:0conv2d_2_11824conv2d_2_11826*
Tin
2*
Tout
2*0
_output_shapes
:€€€€€€€€€

А*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_conv2D_2_layer_call_and_return_conditional_losses_112842"
 conv2D_2/StatefulPartitionedCall£
bn_2/StatefulPartitionedCallStatefulPartitionedCall)conv2D_2/StatefulPartitionedCall:output:0
bn_2_11829
bn_2_11831
bn_2_11833
bn_2_11835*
Tin	
2*
Tout
2*0
_output_shapes
:€€€€€€€€€

А*&
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*H
fCRA
?__inference_bn_2_layer_call_and_return_conditional_losses_116962
bn_2/StatefulPartitionedCallе
activation_2/PartitionedCallPartitionedCall%bn_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*0
_output_shapes
:€€€€€€€€€

А* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*P
fKRI
G__inference_activation_2_layer_call_and_return_conditional_losses_117372
activation_2/PartitionedCallя
max_pool_2/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_max_pool_2_layer_call_and_return_conditional_losses_114262
max_pool_2/PartitionedCallћ
flatten/PartitionedCallPartitionedCall#max_pool_2/PartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_117522
flatten/PartitionedCallл
fc/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0fc_11841fc_11843*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*F
fAR?
=__inference_fc_layer_call_and_return_conditional_losses_117712
fc/StatefulPartitionedCallЏ
IdentityIdentity#fc/StatefulPartitionedCall:output:0^bn_0/StatefulPartitionedCall^bn_1/StatefulPartitionedCall^bn_2/StatefulPartitionedCall!^conv2D_0/StatefulPartitionedCall!^conv2D_1/StatefulPartitionedCall!^conv2D_2/StatefulPartitionedCall^fc/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:€€€€€€€€€@@::::::::::::::::::::2<
bn_0/StatefulPartitionedCallbn_0/StatefulPartitionedCall2<
bn_1/StatefulPartitionedCallbn_1/StatefulPartitionedCall2<
bn_2/StatefulPartitionedCallbn_2/StatefulPartitionedCall2D
 conv2D_0/StatefulPartitionedCall conv2D_0/StatefulPartitionedCall2D
 conv2D_1/StatefulPartitionedCall conv2D_1/StatefulPartitionedCall2D
 conv2D_2/StatefulPartitionedCall conv2D_2/StatefulPartitionedCall28
fc/StatefulPartitionedCallfc/StatefulPartitionedCall:X T
/
_output_shapes
:€€€€€€€€€@@
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
: 
Й
Ч
$__inference_bn_1_layer_call_fn_12772

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallп
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€@*&
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*H
fCRA
?__inference_bn_1_layer_call_and_return_conditional_losses_115932
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€@
 
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
А
C
'__inference_flatten_layer_call_fn_12965

inputs
identityЯ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_117522
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€А:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Н$
∆
?__inference_bn_1_layer_call_and_return_conditional_losses_11575

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ј
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
Const∞
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xњ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub•
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOpё
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/sub_1«
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/mul«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpґ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x«
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЂ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOpк
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/sub_1—
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/mul’
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpЊ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*/
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:€€€€€€€€€@
 
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
”
a
E__inference_activation_layer_call_and_return_conditional_losses_11531

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:€€€€€€€€€@@ 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€@@ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€@@ :W S
/
_output_shapes
:€€€€€€€€€@@ 
 
_user_specified_nameinputs
З
Ч
$__inference_bn_0_layer_call_fn_12587

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallн
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€@@ *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*H
fCRA
?__inference_bn_0_layer_call_and_return_conditional_losses_114722
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€@@ 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€@@ ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€@@ 
 
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
: "ѓL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*≠
serving_defaultЩ
C
input_18
serving_default_input_1:0€€€€€€€€€@@6
fc0
StatefulPartitionedCall:0€€€€€€€€€tensorflow/serving/predict:фе
ўw
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
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
layer_with_weights-4
layer-10
layer_with_weights-5
layer-11
layer-12
layer-13
layer-14
layer_with_weights-6
layer-15
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
ж_default_save_signature
з__call__
+и&call_and_return_all_conditional_losses"€r
_tf_keras_modelеr{"class_name": "Model", "name": "CatDogModel", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "CatDogModel", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "ZeroPadding2D", "config": {"name": "zero_padding2d", "trainable": true, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [2, 2]}, {"class_name": "__tuple__", "items": [2, 2]}]}, "data_format": "channels_last"}, "name": "zero_padding2d", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2D_0", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2D_0", "inbound_nodes": [[["zero_padding2d", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "bn_0", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "bn_0", "inbound_nodes": [[["conv2D_0", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation", "inbound_nodes": [[["bn_0", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pool_0", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pool_0", "inbound_nodes": [[["activation", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2D_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2D_1", "inbound_nodes": [[["max_pool_0", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "bn_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "bn_1", "inbound_nodes": [[["conv2D_1", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_1", "inbound_nodes": [[["bn_1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pool_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pool_1", "inbound_nodes": [[["activation_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2D_2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2D_2", "inbound_nodes": [[["max_pool_1", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "bn_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "bn_2", "inbound_nodes": [[["conv2D_2", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_2", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_2", "inbound_nodes": [[["bn_2", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pool_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pool_2", "inbound_nodes": [[["activation_2", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["max_pool_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "fc", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "fc", "inbound_nodes": [[["flatten", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["fc", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 3]}, "is_graph_network": true, "keras_version": "2.3.0-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "CatDogModel", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "ZeroPadding2D", "config": {"name": "zero_padding2d", "trainable": true, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [2, 2]}, {"class_name": "__tuple__", "items": [2, 2]}]}, "data_format": "channels_last"}, "name": "zero_padding2d", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2D_0", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2D_0", "inbound_nodes": [[["zero_padding2d", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "bn_0", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "bn_0", "inbound_nodes": [[["conv2D_0", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation", "inbound_nodes": [[["bn_0", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pool_0", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pool_0", "inbound_nodes": [[["activation", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2D_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2D_1", "inbound_nodes": [[["max_pool_0", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "bn_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "bn_1", "inbound_nodes": [[["conv2D_1", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_1", "inbound_nodes": [[["bn_1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pool_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pool_1", "inbound_nodes": [[["activation_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2D_2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2D_2", "inbound_nodes": [[["max_pool_1", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "bn_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "bn_2", "inbound_nodes": [[["conv2D_2", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_2", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_2", "inbound_nodes": [[["bn_2", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pool_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pool_2", "inbound_nodes": [[["activation_2", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["max_pool_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "fc", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "fc", "inbound_nodes": [[["flatten", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["fc", 0, 0]]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": ["accuracy"], "weighted_metrics": null, "loss_weights": null, "sample_weight_mode": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
щ"ц
_tf_keras_input_layer÷{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
д
regularization_losses
trainable_variables
	variables
	keras_api
й__call__
+к&call_and_return_all_conditional_losses"”
_tf_keras_layerє{"class_name": "ZeroPadding2D", "name": "zero_padding2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "zero_padding2d", "trainable": true, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [2, 2]}, {"class_name": "__tuple__", "items": [2, 2]}]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
≈	

kernel
bias
regularization_losses
trainable_variables
	variables
 	keras_api
л__call__
+м&call_and_return_all_conditional_losses"Ю
_tf_keras_layerД{"class_name": "Conv2D", "name": "conv2D_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2D_0", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 68, 68, 3]}}
ч
!axis
	"gamma
#beta
$moving_mean
%moving_variance
&regularization_losses
'trainable_variables
(	variables
)	keras_api
н__call__
+о&call_and_return_all_conditional_losses"°
_tf_keras_layerЗ{"class_name": "BatchNormalization", "name": "bn_0", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "bn_0", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 32]}}
∞
*regularization_losses
+trainable_variables
,	variables
-	keras_api
п__call__
+р&call_and_return_all_conditional_losses"Я
_tf_keras_layerЕ{"class_name": "Activation", "name": "activation", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}}
‘
.regularization_losses
/trainable_variables
0	variables
1	keras_api
с__call__
+т&call_and_return_all_conditional_losses"√
_tf_keras_layer©{"class_name": "MaxPooling2D", "name": "max_pool_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "max_pool_0", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
«	

2kernel
3bias
4regularization_losses
5trainable_variables
6	variables
7	keras_api
у__call__
+ф&call_and_return_all_conditional_losses"†
_tf_keras_layerЖ{"class_name": "Conv2D", "name": "conv2D_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2D_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 32]}}
ч
8axis
	9gamma
:beta
;moving_mean
<moving_variance
=regularization_losses
>trainable_variables
?	variables
@	keras_api
х__call__
+ц&call_and_return_all_conditional_losses"°
_tf_keras_layerЗ{"class_name": "BatchNormalization", "name": "bn_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "bn_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 64]}}
і
Aregularization_losses
Btrainable_variables
C	variables
D	keras_api
ч__call__
+ш&call_and_return_all_conditional_losses"£
_tf_keras_layerЙ{"class_name": "Activation", "name": "activation_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}}
‘
Eregularization_losses
Ftrainable_variables
G	variables
H	keras_api
щ__call__
+ъ&call_and_return_all_conditional_losses"√
_tf_keras_layer©{"class_name": "MaxPooling2D", "name": "max_pool_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "max_pool_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
»	

Ikernel
Jbias
Kregularization_losses
Ltrainable_variables
M	variables
N	keras_api
ы__call__
+ь&call_and_return_all_conditional_losses"°
_tf_keras_layerЗ{"class_name": "Conv2D", "name": "conv2D_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2D_2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 14, 14, 64]}}
щ
Oaxis
	Pgamma
Qbeta
Rmoving_mean
Smoving_variance
Tregularization_losses
Utrainable_variables
V	variables
W	keras_api
э__call__
+ю&call_and_return_all_conditional_losses"£
_tf_keras_layerЙ{"class_name": "BatchNormalization", "name": "bn_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "bn_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10, 10, 128]}}
і
Xregularization_losses
Ytrainable_variables
Z	variables
[	keras_api
€__call__
+А&call_and_return_all_conditional_losses"£
_tf_keras_layerЙ{"class_name": "Activation", "name": "activation_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "activation_2", "trainable": true, "dtype": "float32", "activation": "relu"}}
‘
\regularization_losses
]trainable_variables
^	variables
_	keras_api
Б__call__
+В&call_and_return_all_conditional_losses"√
_tf_keras_layer©{"class_name": "MaxPooling2D", "name": "max_pool_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "max_pool_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
Ѕ
`regularization_losses
atrainable_variables
b	variables
c	keras_api
Г__call__
+Д&call_and_return_all_conditional_losses"∞
_tf_keras_layerЦ{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
Ћ

dkernel
ebias
fregularization_losses
gtrainable_variables
h	variables
i	keras_api
Е__call__
+Ж&call_and_return_all_conditional_losses"§
_tf_keras_layerК{"class_name": "Dense", "name": "fc", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "fc", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 3200}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3200]}}
л
jiter

kbeta_1

lbeta_2
	mdecay
nlearning_ratem mЋ"mћ#mЌ2mќ3mѕ9m–:m—Im“Jm”Pm‘Qm’dm÷em„vЎvў"vЏ#vџ2v№3vЁ9vё:vяIvаJvбPvвQvгdvдevе"
	optimizer
 "
trackable_list_wrapper
Ж
0
1
"2
#3
24
35
96
:7
I8
J9
P10
Q11
d12
e13"
trackable_list_wrapper
ґ
0
1
"2
#3
$4
%5
26
37
98
:9
;10
<11
I12
J13
P14
Q15
R16
S17
d18
e19"
trackable_list_wrapper
ќ
regularization_losses
ometrics
trainable_variables
pnon_trainable_variables
	variables

qlayers
rlayer_regularization_losses
slayer_metrics
з__call__
ж_default_save_signature
+и&call_and_return_all_conditional_losses
'и"call_and_return_conditional_losses"
_generic_user_object
-
Зserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
∞
regularization_losses
tmetrics
trainable_variables
unon_trainable_variables
	variables

vlayers
wlayer_regularization_losses
xlayer_metrics
й__call__
+к&call_and_return_all_conditional_losses
'к"call_and_return_conditional_losses"
_generic_user_object
):' 2conv2D_0/kernel
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
∞
regularization_losses
ymetrics
trainable_variables
znon_trainable_variables
	variables

{layers
|layer_regularization_losses
}layer_metrics
л__call__
+м&call_and_return_all_conditional_losses
'м"call_and_return_conditional_losses"
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
≥
&regularization_losses
~metrics
'trainable_variables
non_trainable_variables
(	variables
Аlayers
 Бlayer_regularization_losses
Вlayer_metrics
н__call__
+о&call_and_return_all_conditional_losses
'о"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
*regularization_losses
Гmetrics
+trainable_variables
Дnon_trainable_variables
,	variables
Еlayers
 Жlayer_regularization_losses
Зlayer_metrics
п__call__
+р&call_and_return_all_conditional_losses
'р"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
.regularization_losses
Иmetrics
/trainable_variables
Йnon_trainable_variables
0	variables
Кlayers
 Лlayer_regularization_losses
Мlayer_metrics
с__call__
+т&call_and_return_all_conditional_losses
'т"call_and_return_conditional_losses"
_generic_user_object
):' @2conv2D_1/kernel
:@2conv2D_1/bias
 "
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
µ
4regularization_losses
Нmetrics
5trainable_variables
Оnon_trainable_variables
6	variables
Пlayers
 Рlayer_regularization_losses
Сlayer_metrics
у__call__
+ф&call_and_return_all_conditional_losses
'ф"call_and_return_conditional_losses"
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
90
:1"
trackable_list_wrapper
<
90
:1
;2
<3"
trackable_list_wrapper
µ
=regularization_losses
Тmetrics
>trainable_variables
Уnon_trainable_variables
?	variables
Фlayers
 Хlayer_regularization_losses
Цlayer_metrics
х__call__
+ц&call_and_return_all_conditional_losses
'ц"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Aregularization_losses
Чmetrics
Btrainable_variables
Шnon_trainable_variables
C	variables
Щlayers
 Ъlayer_regularization_losses
Ыlayer_metrics
ч__call__
+ш&call_and_return_all_conditional_losses
'ш"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Eregularization_losses
Ьmetrics
Ftrainable_variables
Эnon_trainable_variables
G	variables
Юlayers
 Яlayer_regularization_losses
†layer_metrics
щ__call__
+ъ&call_and_return_all_conditional_losses
'ъ"call_and_return_conditional_losses"
_generic_user_object
*:(@А2conv2D_2/kernel
:А2conv2D_2/bias
 "
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
µ
Kregularization_losses
°metrics
Ltrainable_variables
Ґnon_trainable_variables
M	variables
£layers
 §layer_regularization_losses
•layer_metrics
ы__call__
+ь&call_and_return_all_conditional_losses
'ь"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
:А2
bn_2/gamma
:А2	bn_2/beta
!:А (2bn_2/moving_mean
%:#А (2bn_2/moving_variance
 "
trackable_list_wrapper
.
P0
Q1"
trackable_list_wrapper
<
P0
Q1
R2
S3"
trackable_list_wrapper
µ
Tregularization_losses
¶metrics
Utrainable_variables
Іnon_trainable_variables
V	variables
®layers
 ©layer_regularization_losses
™layer_metrics
э__call__
+ю&call_and_return_all_conditional_losses
'ю"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Xregularization_losses
Ђmetrics
Ytrainable_variables
ђnon_trainable_variables
Z	variables
≠layers
 Ѓlayer_regularization_losses
ѓlayer_metrics
€__call__
+А&call_and_return_all_conditional_losses
'А"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
\regularization_losses
∞metrics
]trainable_variables
±non_trainable_variables
^	variables
≤layers
 ≥layer_regularization_losses
іlayer_metrics
Б__call__
+В&call_and_return_all_conditional_losses
'В"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
`regularization_losses
µmetrics
atrainable_variables
ґnon_trainable_variables
b	variables
Јlayers
 Єlayer_regularization_losses
єlayer_metrics
Г__call__
+Д&call_and_return_all_conditional_losses
'Д"call_and_return_conditional_losses"
_generic_user_object
:	А2	fc/kernel
:2fc/bias
 "
trackable_list_wrapper
.
d0
e1"
trackable_list_wrapper
.
d0
e1"
trackable_list_wrapper
µ
fregularization_losses
Їmetrics
gtrainable_variables
їnon_trainable_variables
h	variables
Љlayers
 љlayer_regularization_losses
Њlayer_metrics
Е__call__
+Ж&call_and_return_all_conditional_losses
'Ж"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
0
њ0
ј1"
trackable_list_wrapper
J
$0
%1
;2
<3
R4
S5"
trackable_list_wrapper
Ц
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
15"
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
.
;0
<1"
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
R0
S1"
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
њ

Ѕtotal

¬count
√	variables
ƒ	keras_api"Д
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
Д

≈total

∆count
«
_fn_kwargs
»	variables
…	keras_api"Є
_tf_keras_metricЭ{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}
:  (2total
:  (2count
0
Ѕ0
¬1"
trackable_list_wrapper
.
√	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
≈0
∆1"
trackable_list_wrapper
.
»	variables"
_generic_user_object
.:, 2Adam/conv2D_0/kernel/m
 : 2Adam/conv2D_0/bias/m
: 2Adam/bn_0/gamma/m
: 2Adam/bn_0/beta/m
.:, @2Adam/conv2D_1/kernel/m
 :@2Adam/conv2D_1/bias/m
:@2Adam/bn_1/gamma/m
:@2Adam/bn_1/beta/m
/:-@А2Adam/conv2D_2/kernel/m
!:А2Adam/conv2D_2/bias/m
:А2Adam/bn_2/gamma/m
:А2Adam/bn_2/beta/m
!:	А2Adam/fc/kernel/m
:2Adam/fc/bias/m
.:, 2Adam/conv2D_0/kernel/v
 : 2Adam/conv2D_0/bias/v
: 2Adam/bn_0/gamma/v
: 2Adam/bn_0/beta/v
.:, @2Adam/conv2D_1/kernel/v
 :@2Adam/conv2D_1/bias/v
:@2Adam/bn_1/gamma/v
:@2Adam/bn_1/beta/v
/:-@А2Adam/conv2D_2/kernel/v
!:А2Adam/conv2D_2/bias/v
:А2Adam/bn_2/gamma/v
:А2Adam/bn_2/beta/v
!:	А2Adam/fc/kernel/v
:2Adam/fc/bias/v
ж2г
 __inference__wrapped_model_10942Њ
Л≤З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *.Ґ+
)К&
input_1€€€€€€€€€@@
ъ2ч
+__inference_CatDogModel_layer_call_fn_12393
+__inference_CatDogModel_layer_call_fn_12438
+__inference_CatDogModel_layer_call_fn_12056
+__inference_CatDogModel_layer_call_fn_11952ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ж2г
F__inference_CatDogModel_layer_call_and_return_conditional_losses_12267
F__inference_CatDogModel_layer_call_and_return_conditional_losses_11847
F__inference_CatDogModel_layer_call_and_return_conditional_losses_12348
F__inference_CatDogModel_layer_call_and_return_conditional_losses_11788ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ц2У
.__inference_zero_padding2d_layer_call_fn_10955а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
±2Ѓ
I__inference_zero_padding2d_layer_call_and_return_conditional_losses_10949а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
З2Д
(__inference_conv2D_0_layer_call_fn_10976„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ґ2Я
C__inference_conv2D_0_layer_call_and_return_conditional_losses_10966„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€
“2ѕ
$__inference_bn_0_layer_call_fn_12525
$__inference_bn_0_layer_call_fn_12512
$__inference_bn_0_layer_call_fn_12600
$__inference_bn_0_layer_call_fn_12587і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Њ2ї
?__inference_bn_0_layer_call_and_return_conditional_losses_12574
?__inference_bn_0_layer_call_and_return_conditional_losses_12499
?__inference_bn_0_layer_call_and_return_conditional_losses_12556
?__inference_bn_0_layer_call_and_return_conditional_losses_12481і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
‘2—
*__inference_activation_layer_call_fn_12610Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
п2м
E__inference_activation_layer_call_and_return_conditional_losses_12605Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Т2П
*__inference_max_pool_0_layer_call_fn_11114а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
≠2™
E__inference_max_pool_0_layer_call_and_return_conditional_losses_11108а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
З2Д
(__inference_conv2D_1_layer_call_fn_11135„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
Ґ2Я
C__inference_conv2D_1_layer_call_and_return_conditional_losses_11125„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
“2ѕ
$__inference_bn_1_layer_call_fn_12684
$__inference_bn_1_layer_call_fn_12697
$__inference_bn_1_layer_call_fn_12772
$__inference_bn_1_layer_call_fn_12759і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Њ2ї
?__inference_bn_1_layer_call_and_return_conditional_losses_12728
?__inference_bn_1_layer_call_and_return_conditional_losses_12671
?__inference_bn_1_layer_call_and_return_conditional_losses_12653
?__inference_bn_1_layer_call_and_return_conditional_losses_12746і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
÷2”
,__inference_activation_1_layer_call_fn_12782Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
с2о
G__inference_activation_1_layer_call_and_return_conditional_losses_12777Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Т2П
*__inference_max_pool_1_layer_call_fn_11273а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
≠2™
E__inference_max_pool_1_layer_call_and_return_conditional_losses_11267а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
З2Д
(__inference_conv2D_2_layer_call_fn_11294„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ґ2Я
C__inference_conv2D_2_layer_call_and_return_conditional_losses_11284„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
“2ѕ
$__inference_bn_2_layer_call_fn_12869
$__inference_bn_2_layer_call_fn_12944
$__inference_bn_2_layer_call_fn_12856
$__inference_bn_2_layer_call_fn_12931і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Њ2ї
?__inference_bn_2_layer_call_and_return_conditional_losses_12900
?__inference_bn_2_layer_call_and_return_conditional_losses_12825
?__inference_bn_2_layer_call_and_return_conditional_losses_12918
?__inference_bn_2_layer_call_and_return_conditional_losses_12843і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
÷2”
,__inference_activation_2_layer_call_fn_12954Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
с2о
G__inference_activation_2_layer_call_and_return_conditional_losses_12949Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Т2П
*__inference_max_pool_2_layer_call_fn_11432а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
≠2™
E__inference_max_pool_2_layer_call_and_return_conditional_losses_11426а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
—2ќ
'__inference_flatten_layer_call_fn_12965Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
м2й
B__inference_flatten_layer_call_and_return_conditional_losses_12960Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ћ2…
"__inference_fc_layer_call_fn_12985Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
з2д
=__inference_fc_layer_call_and_return_conditional_losses_12976Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
2B0
#__inference_signature_wrapper_12147input_1…
F__inference_CatDogModel_layer_call_and_return_conditional_losses_11788"#$%239:;<IJPQRSde@Ґ=
6Ґ3
)К&
input_1€€€€€€€€€@@
p

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ …
F__inference_CatDogModel_layer_call_and_return_conditional_losses_11847"#$%239:;<IJPQRSde@Ґ=
6Ґ3
)К&
input_1€€€€€€€€€@@
p 

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ »
F__inference_CatDogModel_layer_call_and_return_conditional_losses_12267~"#$%239:;<IJPQRSde?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€@@
p

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ »
F__inference_CatDogModel_layer_call_and_return_conditional_losses_12348~"#$%239:;<IJPQRSde?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€@@
p 

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ °
+__inference_CatDogModel_layer_call_fn_11952r"#$%239:;<IJPQRSde@Ґ=
6Ґ3
)К&
input_1€€€€€€€€€@@
p

 
™ "К€€€€€€€€€°
+__inference_CatDogModel_layer_call_fn_12056r"#$%239:;<IJPQRSde@Ґ=
6Ґ3
)К&
input_1€€€€€€€€€@@
p 

 
™ "К€€€€€€€€€†
+__inference_CatDogModel_layer_call_fn_12393q"#$%239:;<IJPQRSde?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€@@
p

 
™ "К€€€€€€€€€†
+__inference_CatDogModel_layer_call_fn_12438q"#$%239:;<IJPQRSde?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€@@
p 

 
™ "К€€€€€€€€€Э
 __inference__wrapped_model_10942y"#$%239:;<IJPQRSde8Ґ5
.Ґ+
)К&
input_1€€€€€€€€€@@
™ "'™$
"
fcК
fc€€€€€€€€€≥
G__inference_activation_1_layer_call_and_return_conditional_losses_12777h7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€@
™ "-Ґ*
#К 
0€€€€€€€€€@
Ъ Л
,__inference_activation_1_layer_call_fn_12782[7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€@
™ " К€€€€€€€€€@µ
G__inference_activation_2_layer_call_and_return_conditional_losses_12949j8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€

А
™ ".Ґ+
$К!
0€€€€€€€€€

А
Ъ Н
,__inference_activation_2_layer_call_fn_12954]8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€

А
™ "!К€€€€€€€€€

А±
E__inference_activation_layer_call_and_return_conditional_losses_12605h7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€@@ 
™ "-Ґ*
#К 
0€€€€€€€€€@@ 
Ъ Й
*__inference_activation_layer_call_fn_12610[7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€@@ 
™ " К€€€€€€€€€@@ Џ
?__inference_bn_0_layer_call_and_return_conditional_losses_12481Ц"#$%MҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
p
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
Ъ Џ
?__inference_bn_0_layer_call_and_return_conditional_losses_12499Ц"#$%MҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
p 
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
Ъ µ
?__inference_bn_0_layer_call_and_return_conditional_losses_12556r"#$%;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€@@ 
p
™ "-Ґ*
#К 
0€€€€€€€€€@@ 
Ъ µ
?__inference_bn_0_layer_call_and_return_conditional_losses_12574r"#$%;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€@@ 
p 
™ "-Ґ*
#К 
0€€€€€€€€€@@ 
Ъ ≤
$__inference_bn_0_layer_call_fn_12512Й"#$%MҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
p
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€ ≤
$__inference_bn_0_layer_call_fn_12525Й"#$%MҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
p 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€ Н
$__inference_bn_0_layer_call_fn_12587e"#$%;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€@@ 
p
™ " К€€€€€€€€€@@ Н
$__inference_bn_0_layer_call_fn_12600e"#$%;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€@@ 
p 
™ " К€€€€€€€€€@@ Џ
?__inference_bn_1_layer_call_and_return_conditional_losses_12653Ц9:;<MҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ Џ
?__inference_bn_1_layer_call_and_return_conditional_losses_12671Ц9:;<MҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p 
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ µ
?__inference_bn_1_layer_call_and_return_conditional_losses_12728r9:;<;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€@
p
™ "-Ґ*
#К 
0€€€€€€€€€@
Ъ µ
?__inference_bn_1_layer_call_and_return_conditional_losses_12746r9:;<;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€@
p 
™ "-Ґ*
#К 
0€€€€€€€€€@
Ъ ≤
$__inference_bn_1_layer_call_fn_12684Й9:;<MҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@≤
$__inference_bn_1_layer_call_fn_12697Й9:;<MҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@Н
$__inference_bn_1_layer_call_fn_12759e9:;<;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€@
p
™ " К€€€€€€€€€@Н
$__inference_bn_1_layer_call_fn_12772e9:;<;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€@
p 
™ " К€€€€€€€€€@№
?__inference_bn_2_layer_call_and_return_conditional_losses_12825ШPQRSNҐK
DҐA
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
p
™ "@Ґ=
6К3
0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Ъ №
?__inference_bn_2_layer_call_and_return_conditional_losses_12843ШPQRSNҐK
DҐA
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
p 
™ "@Ґ=
6К3
0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Ъ Ј
?__inference_bn_2_layer_call_and_return_conditional_losses_12900tPQRS<Ґ9
2Ґ/
)К&
inputs€€€€€€€€€

А
p
™ ".Ґ+
$К!
0€€€€€€€€€

А
Ъ Ј
?__inference_bn_2_layer_call_and_return_conditional_losses_12918tPQRS<Ґ9
2Ґ/
)К&
inputs€€€€€€€€€

А
p 
™ ".Ґ+
$К!
0€€€€€€€€€

А
Ъ і
$__inference_bn_2_layer_call_fn_12856ЛPQRSNҐK
DҐA
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
p
™ "3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€Аі
$__inference_bn_2_layer_call_fn_12869ЛPQRSNҐK
DҐA
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
p 
™ "3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€АП
$__inference_bn_2_layer_call_fn_12931gPQRS<Ґ9
2Ґ/
)К&
inputs€€€€€€€€€

А
p
™ "!К€€€€€€€€€

АП
$__inference_bn_2_layer_call_fn_12944gPQRS<Ґ9
2Ґ/
)К&
inputs€€€€€€€€€

А
p 
™ "!К€€€€€€€€€

АЎ
C__inference_conv2D_0_layer_call_and_return_conditional_losses_10966РIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
Ъ ∞
(__inference_conv2D_0_layer_call_fn_10976ГIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€ Ў
C__inference_conv2D_1_layer_call_and_return_conditional_losses_11125Р23IҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ ∞
(__inference_conv2D_1_layer_call_fn_11135Г23IҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@ў
C__inference_conv2D_2_layer_call_and_return_conditional_losses_11284СIJIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ "@Ґ=
6К3
0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Ъ ±
(__inference_conv2D_2_layer_call_fn_11294ДIJIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ "3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€АЮ
=__inference_fc_layer_call_and_return_conditional_losses_12976]de0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "%Ґ"
К
0€€€€€€€€€
Ъ v
"__inference_fc_layer_call_fn_12985Pde0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "К€€€€€€€€€®
B__inference_flatten_layer_call_and_return_conditional_losses_12960b8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€А
™ "&Ґ#
К
0€€€€€€€€€А
Ъ А
'__inference_flatten_layer_call_fn_12965U8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€А
™ "К€€€€€€€€€Аи
E__inference_max_pool_0_layer_call_and_return_conditional_losses_11108ЮRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ј
*__inference_max_pool_0_layer_call_fn_11114СRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€и
E__inference_max_pool_1_layer_call_and_return_conditional_losses_11267ЮRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ј
*__inference_max_pool_1_layer_call_fn_11273СRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€и
E__inference_max_pool_2_layer_call_and_return_conditional_losses_11426ЮRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ј
*__inference_max_pool_2_layer_call_fn_11432СRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€ђ
#__inference_signature_wrapper_12147Д"#$%239:;<IJPQRSdeCҐ@
Ґ 
9™6
4
input_1)К&
input_1€€€€€€€€€@@"'™$
"
fcК
fc€€€€€€€€€м
I__inference_zero_padding2d_layer_call_and_return_conditional_losses_10949ЮRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ƒ
.__inference_zero_padding2d_layer_call_fn_10955СRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€