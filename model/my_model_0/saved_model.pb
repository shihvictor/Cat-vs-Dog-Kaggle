��	
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
shapeshape�"serve*2.2.02v2.2.0-rc4-8-g2b96f3662b8ԧ
|
conv0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv0/kernel
u
 conv0/kernel/Read/ReadVariableOpReadVariableOpconv0/kernel*&
_output_shapes
: *
dtype0
l

conv0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
conv0/bias
e
conv0/bias/Read/ReadVariableOpReadVariableOp
conv0/bias*
_output_shapes
: *
dtype0
j
	bn0/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	bn0/gamma
c
bn0/gamma/Read/ReadVariableOpReadVariableOp	bn0/gamma*
_output_shapes
: *
dtype0
h
bn0/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
bn0/beta
a
bn0/beta/Read/ReadVariableOpReadVariableOpbn0/beta*
_output_shapes
: *
dtype0
v
bn0/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_namebn0/moving_mean
o
#bn0/moving_mean/Read/ReadVariableOpReadVariableOpbn0/moving_mean*
_output_shapes
: *
dtype0
~
bn0/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_namebn0/moving_variance
w
'bn0/moving_variance/Read/ReadVariableOpReadVariableOpbn0/moving_variance*
_output_shapes
: *
dtype0
p
	fc/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*
shared_name	fc/kernel
i
fc/kernel/Read/ReadVariableOpReadVariableOp	fc/kernel* 
_output_shapes
:
��*
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
�
Adam/conv0/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/conv0/kernel/m
�
'Adam/conv0/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv0/kernel/m*&
_output_shapes
: *
dtype0
z
Adam/conv0/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameAdam/conv0/bias/m
s
%Adam/conv0/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv0/bias/m*
_output_shapes
: *
dtype0
x
Adam/bn0/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameAdam/bn0/gamma/m
q
$Adam/bn0/gamma/m/Read/ReadVariableOpReadVariableOpAdam/bn0/gamma/m*
_output_shapes
: *
dtype0
v
Adam/bn0/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameAdam/bn0/beta/m
o
#Adam/bn0/beta/m/Read/ReadVariableOpReadVariableOpAdam/bn0/beta/m*
_output_shapes
: *
dtype0
~
Adam/fc/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*!
shared_nameAdam/fc/kernel/m
w
$Adam/fc/kernel/m/Read/ReadVariableOpReadVariableOpAdam/fc/kernel/m* 
_output_shapes
:
��*
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
�
Adam/conv0/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/conv0/kernel/v
�
'Adam/conv0/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv0/kernel/v*&
_output_shapes
: *
dtype0
z
Adam/conv0/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameAdam/conv0/bias/v
s
%Adam/conv0/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv0/bias/v*
_output_shapes
: *
dtype0
x
Adam/bn0/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameAdam/bn0/gamma/v
q
$Adam/bn0/gamma/v/Read/ReadVariableOpReadVariableOpAdam/bn0/gamma/v*
_output_shapes
: *
dtype0
v
Adam/bn0/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameAdam/bn0/beta/v
o
#Adam/bn0/beta/v/Read/ReadVariableOpReadVariableOpAdam/bn0/beta/v*
_output_shapes
: *
dtype0
~
Adam/fc/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*!
shared_nameAdam/fc/kernel/v
w
$Adam/fc/kernel/v/Read/ReadVariableOpReadVariableOpAdam/fc/kernel/v* 
_output_shapes
:
��*
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
�0
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�/
value�/B�/ B�/
�
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer-6
layer_with_weights-2
layer-7
		optimizer

	variables
trainable_variables
regularization_losses
	keras_api

signatures
 
R
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
�
axis
	gamma
beta
moving_mean
moving_variance
	variables
trainable_variables
 regularization_losses
!	keras_api
R
"	variables
#trainable_variables
$regularization_losses
%	keras_api
R
&	variables
'trainable_variables
(regularization_losses
)	keras_api
R
*	variables
+trainable_variables
,regularization_losses
-	keras_api
h

.kernel
/bias
0	variables
1trainable_variables
2regularization_losses
3	keras_api
�
4iter

5beta_1

6beta_2
	7decay
8learning_ratemlmmmnmo.mp/mqvrvsvtvu.vv/vw
8
0
1
2
3
4
5
.6
/7
*
0
1
2
3
.4
/5
 
�
9layer_metrics
:non_trainable_variables

;layers
<layer_regularization_losses

	variables
trainable_variables
=metrics
regularization_losses
 
 
 
 
�
>layer_metrics
?non_trainable_variables

@layers
Alayer_regularization_losses
	variables
trainable_variables
Bmetrics
regularization_losses
XV
VARIABLE_VALUEconv0/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
conv0/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
�
Clayer_metrics
Dnon_trainable_variables

Elayers
Flayer_regularization_losses
	variables
trainable_variables
Gmetrics
regularization_losses
 
TR
VARIABLE_VALUE	bn0/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEbn0/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEbn0/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUEbn0/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

0
1
2
3

0
1
 
�
Hlayer_metrics
Inon_trainable_variables

Jlayers
Klayer_regularization_losses
	variables
trainable_variables
Lmetrics
 regularization_losses
 
 
 
�
Mlayer_metrics
Nnon_trainable_variables

Olayers
Player_regularization_losses
"	variables
#trainable_variables
Qmetrics
$regularization_losses
 
 
 
�
Rlayer_metrics
Snon_trainable_variables

Tlayers
Ulayer_regularization_losses
&	variables
'trainable_variables
Vmetrics
(regularization_losses
 
 
 
�
Wlayer_metrics
Xnon_trainable_variables

Ylayers
Zlayer_regularization_losses
*	variables
+trainable_variables
[metrics
,regularization_losses
US
VARIABLE_VALUE	fc/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEfc/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

.0
/1

.0
/1
 
�
\layer_metrics
]non_trainable_variables

^layers
_layer_regularization_losses
0	variables
1trainable_variables
`metrics
2regularization_losses
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
 

0
1
8
0
1
2
3
4
5
6
7
 

a0
b1
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
0
1
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
4
	ctotal
	dcount
e	variables
f	keras_api
D
	gtotal
	hcount
i
_fn_kwargs
j	variables
k	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

c0
d1

e	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

g0
h1

j	variables
{y
VARIABLE_VALUEAdam/conv0/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv0/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/bn0/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/bn0/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/fc/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/fc/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv0/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv0/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/bn0/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/bn0/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/fc/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/fc/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�
serving_default_input_1Placeholder*/
_output_shapes
:���������@@*
dtype0*$
shape:���������@@
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv0/kernel
conv0/bias	bn0/gammabn0/betabn0/moving_meanbn0/moving_variance	fc/kernelfc/bias*
Tin
2	*
Tout
2*'
_output_shapes
:���������**
_read_only_resource_inputs

**
config_proto

GPU 

CPU2J 8*+
f&R$
"__inference_signature_wrapper_1771
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename conv0/kernel/Read/ReadVariableOpconv0/bias/Read/ReadVariableOpbn0/gamma/Read/ReadVariableOpbn0/beta/Read/ReadVariableOp#bn0/moving_mean/Read/ReadVariableOp'bn0/moving_variance/Read/ReadVariableOpfc/kernel/Read/ReadVariableOpfc/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp'Adam/conv0/kernel/m/Read/ReadVariableOp%Adam/conv0/bias/m/Read/ReadVariableOp$Adam/bn0/gamma/m/Read/ReadVariableOp#Adam/bn0/beta/m/Read/ReadVariableOp$Adam/fc/kernel/m/Read/ReadVariableOp"Adam/fc/bias/m/Read/ReadVariableOp'Adam/conv0/kernel/v/Read/ReadVariableOp%Adam/conv0/bias/v/Read/ReadVariableOp$Adam/bn0/gamma/v/Read/ReadVariableOp#Adam/bn0/beta/v/Read/ReadVariableOp$Adam/fc/kernel/v/Read/ReadVariableOp"Adam/fc/bias/v/Read/ReadVariableOpConst**
Tin#
!2	*
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
CPU2J 8*&
f!R
__inference__traced_save_2217
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv0/kernel
conv0/bias	bn0/gammabn0/betabn0/moving_meanbn0/moving_variance	fc/kernelfc/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/conv0/kernel/mAdam/conv0/bias/mAdam/bn0/gamma/mAdam/bn0/beta/mAdam/fc/kernel/mAdam/fc/bias/mAdam/conv0/kernel/vAdam/conv0/bias/vAdam/bn0/gamma/vAdam/bn0/beta/vAdam/fc/kernel/vAdam/fc/bias/v*)
Tin"
 2*
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
CPU2J 8*)
f$R"
 __inference__traced_restore_2316�
�
d
H__inference_zero_padding2d_layer_call_and_return_conditional_losses_1289

inputs
identity�
Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             2
Pad/paddings�
PadPadinputsPad/paddings:output:0*
T0*J
_output_shapes8
6:4������������������������������������2
Pad�
IdentityIdentityPad:output:0*
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
�
E__inference_CatDogModel_layer_call_and_return_conditional_losses_1631
input_1

conv0_1608

conv0_1610
bn0_1613
bn0_1615
bn0_1617
bn0_1619
fc_1625
fc_1627
identity��bn0/StatefulPartitionedCall�conv0/StatefulPartitionedCall�fc/StatefulPartitionedCall�
zero_padding2d/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*/
_output_shapes
:���������FF* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*Q
fLRJ
H__inference_zero_padding2d_layer_call_and_return_conditional_losses_12892 
zero_padding2d/PartitionedCall�
conv0/StatefulPartitionedCallStatefulPartitionedCall'zero_padding2d/PartitionedCall:output:0
conv0_1608
conv0_1610*
Tin
2*
Tout
2*/
_output_shapes
:���������@@ *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*H
fCRA
?__inference_conv0_layer_call_and_return_conditional_losses_13062
conv0/StatefulPartitionedCall�
bn0/StatefulPartitionedCallStatefulPartitionedCall&conv0/StatefulPartitionedCall:output:0bn0_1613bn0_1615bn0_1617bn0_1619*
Tin	
2*
Tout
2*/
_output_shapes
:���������@@ *&
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*F
fAR?
=__inference_bn0_layer_call_and_return_conditional_losses_15122
bn0/StatefulPartitionedCall�
activation/PartitionedCallPartitionedCall$bn0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:���������@@ * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_activation_layer_call_and_return_conditional_losses_15532
activation/PartitionedCall�
max_pool0/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:���������   * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_max_pool0_layer_call_and_return_conditional_losses_14482
max_pool0/PartitionedCall�
flatten/PartitionedCallPartitionedCall"max_pool0/PartitionedCall:output:0*
Tin
2*
Tout
2*)
_output_shapes
:�����������* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_15682
flatten/PartitionedCall�
fc/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0fc_1625fc_1627*
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
GPU 

CPU2J 8*E
f@R>
<__inference_fc_layer_call_and_return_conditional_losses_15872
fc/StatefulPartitionedCall�
IdentityIdentity#fc/StatefulPartitionedCall:output:0^bn0/StatefulPartitionedCall^conv0/StatefulPartitionedCall^fc/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:���������@@::::::::2:
bn0/StatefulPartitionedCallbn0/StatefulPartitionedCall2>
conv0/StatefulPartitionedCallconv0/StatefulPartitionedCall28
fc/StatefulPartitionedCallfc/StatefulPartitionedCall:X T
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
: 
�	
�
?__inference_conv0_layer_call_and_return_conditional_losses_1306

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
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
]
A__inference_flatten_layer_call_and_return_conditional_losses_1568

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"���� �  2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:�����������2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������   :W S
/
_output_shapes
:���������   
 
_user_specified_nameinputs
�
�
E__inference_CatDogModel_layer_call_and_return_conditional_losses_1709

inputs

conv0_1686

conv0_1688
bn0_1691
bn0_1693
bn0_1695
bn0_1697
fc_1703
fc_1705
identity��bn0/StatefulPartitionedCall�conv0/StatefulPartitionedCall�fc/StatefulPartitionedCall�
zero_padding2d/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*/
_output_shapes
:���������FF* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*Q
fLRJ
H__inference_zero_padding2d_layer_call_and_return_conditional_losses_12892 
zero_padding2d/PartitionedCall�
conv0/StatefulPartitionedCallStatefulPartitionedCall'zero_padding2d/PartitionedCall:output:0
conv0_1686
conv0_1688*
Tin
2*
Tout
2*/
_output_shapes
:���������@@ *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*H
fCRA
?__inference_conv0_layer_call_and_return_conditional_losses_13062
conv0/StatefulPartitionedCall�
bn0/StatefulPartitionedCallStatefulPartitionedCall&conv0/StatefulPartitionedCall:output:0bn0_1691bn0_1693bn0_1695bn0_1697*
Tin	
2*
Tout
2*/
_output_shapes
:���������@@ *&
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*F
fAR?
=__inference_bn0_layer_call_and_return_conditional_losses_15122
bn0/StatefulPartitionedCall�
activation/PartitionedCallPartitionedCall$bn0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:���������@@ * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_activation_layer_call_and_return_conditional_losses_15532
activation/PartitionedCall�
max_pool0/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:���������   * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_max_pool0_layer_call_and_return_conditional_losses_14482
max_pool0/PartitionedCall�
flatten/PartitionedCallPartitionedCall"max_pool0/PartitionedCall:output:0*
Tin
2*
Tout
2*)
_output_shapes
:�����������* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_15682
flatten/PartitionedCall�
fc/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0fc_1703fc_1705*
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
GPU 

CPU2J 8*E
f@R>
<__inference_fc_layer_call_and_return_conditional_losses_15872
fc/StatefulPartitionedCall�
IdentityIdentity#fc/StatefulPartitionedCall:output:0^bn0/StatefulPartitionedCall^conv0/StatefulPartitionedCall^fc/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:���������@@::::::::2:
bn0/StatefulPartitionedCallbn0/StatefulPartitionedCall2>
conv0/StatefulPartitionedCallconv0/StatefulPartitionedCall28
fc/StatefulPartitionedCallfc/StatefulPartitionedCall:W S
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
: 
�	
�
"__inference_signature_wrapper_1771
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*'
_output_shapes
:���������**
_read_only_resource_inputs

**
config_proto

GPU 

CPU2J 8*(
f#R!
__inference__wrapped_model_12822
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:���������@@::::::::22
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
: 
�
_
C__inference_max_pool0_layer_call_and_return_conditional_losses_1448

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
"__inference_bn0_layer_call_fn_2062

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
-:+��������������������������� *&
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*F
fAR?
=__inference_bn0_layer_call_and_return_conditional_losses_14312
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
�
`
D__inference_activation_layer_call_and_return_conditional_losses_1553

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:���������@@ 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������@@ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@@ :W S
/
_output_shapes
:���������@@ 
 
_user_specified_nameinputs
�!
�
E__inference_CatDogModel_layer_call_and_return_conditional_losses_1858

inputs(
$conv0_conv2d_readvariableop_resource)
%conv0_biasadd_readvariableop_resource
bn0_readvariableop_resource!
bn0_readvariableop_1_resource0
,bn0_fusedbatchnormv3_readvariableop_resource2
.bn0_fusedbatchnormv3_readvariableop_1_resource%
!fc_matmul_readvariableop_resource&
"fc_biasadd_readvariableop_resource
identity��
zero_padding2d/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             2
zero_padding2d/Pad/paddings�
zero_padding2d/PadPadinputs$zero_padding2d/Pad/paddings:output:0*
T0*/
_output_shapes
:���������FF2
zero_padding2d/Pad�
conv0/Conv2D/ReadVariableOpReadVariableOp$conv0_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv0/Conv2D/ReadVariableOp�
conv0/Conv2DConv2Dzero_padding2d/Pad:output:0#conv0/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ *
paddingVALID*
strides
2
conv0/Conv2D�
conv0/BiasAdd/ReadVariableOpReadVariableOp%conv0_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv0/BiasAdd/ReadVariableOp�
conv0/BiasAddBiasAddconv0/Conv2D:output:0$conv0/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ 2
conv0/BiasAdd�
bn0/ReadVariableOpReadVariableOpbn0_readvariableop_resource*
_output_shapes
: *
dtype02
bn0/ReadVariableOp�
bn0/ReadVariableOp_1ReadVariableOpbn0_readvariableop_1_resource*
_output_shapes
: *
dtype02
bn0/ReadVariableOp_1�
#bn0/FusedBatchNormV3/ReadVariableOpReadVariableOp,bn0_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02%
#bn0/FusedBatchNormV3/ReadVariableOp�
%bn0/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp.bn0_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02'
%bn0/FusedBatchNormV3/ReadVariableOp_1�
bn0/FusedBatchNormV3FusedBatchNormV3conv0/BiasAdd:output:0bn0/ReadVariableOp:value:0bn0/ReadVariableOp_1:value:0+bn0/FusedBatchNormV3/ReadVariableOp:value:0-bn0/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@@ : : : : :*
epsilon%o�:*
is_training( 2
bn0/FusedBatchNormV3~
activation/ReluRelubn0/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������@@ 2
activation/Relu�
max_pool0/MaxPoolMaxPoolactivation/Relu:activations:0*/
_output_shapes
:���������   *
ksize
*
paddingVALID*
strides
2
max_pool0/MaxPoolo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� �  2
flatten/Const�
flatten/ReshapeReshapemax_pool0/MaxPool:output:0flatten/Const:output:0*
T0*)
_output_shapes
:�����������2
flatten/Reshape�
fc/MatMul/ReadVariableOpReadVariableOp!fc_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
fc/MatMul/ReadVariableOp�
	fc/MatMulMatMulflatten/Reshape:output:0 fc/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
	fc/MatMul�
fc/BiasAdd/ReadVariableOpReadVariableOp"fc_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
fc/BiasAdd/ReadVariableOp�

fc/BiasAddBiasAddfc/MatMul:product:0!fc/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2

fc/BiasAddj

fc/SoftmaxSoftmaxfc/BiasAdd:output:0*
T0*'
_output_shapes
:���������2

fc/Softmaxh
IdentityIdentityfc/Softmax:softmax:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:���������@@:::::::::W S
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
: 
�
`
D__inference_activation_layer_call_and_return_conditional_losses_2067

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:���������@@ 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������@@ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@@ :W S
/
_output_shapes
:���������@@ 
 
_user_specified_nameinputs
�$
�
=__inference_bn0_layer_call_and_return_conditional_losses_1943

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
7:���������@@ : : : : :*
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
:���������@@ 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@@ ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:���������@@ 
 
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
�(
�
__inference__wrapped_model_1282
input_14
0catdogmodel_conv0_conv2d_readvariableop_resource5
1catdogmodel_conv0_biasadd_readvariableop_resource+
'catdogmodel_bn0_readvariableop_resource-
)catdogmodel_bn0_readvariableop_1_resource<
8catdogmodel_bn0_fusedbatchnormv3_readvariableop_resource>
:catdogmodel_bn0_fusedbatchnormv3_readvariableop_1_resource1
-catdogmodel_fc_matmul_readvariableop_resource2
.catdogmodel_fc_biasadd_readvariableop_resource
identity��
'CatDogModel/zero_padding2d/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             2)
'CatDogModel/zero_padding2d/Pad/paddings�
CatDogModel/zero_padding2d/PadPadinput_10CatDogModel/zero_padding2d/Pad/paddings:output:0*
T0*/
_output_shapes
:���������FF2 
CatDogModel/zero_padding2d/Pad�
'CatDogModel/conv0/Conv2D/ReadVariableOpReadVariableOp0catdogmodel_conv0_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02)
'CatDogModel/conv0/Conv2D/ReadVariableOp�
CatDogModel/conv0/Conv2DConv2D'CatDogModel/zero_padding2d/Pad:output:0/CatDogModel/conv0/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ *
paddingVALID*
strides
2
CatDogModel/conv0/Conv2D�
(CatDogModel/conv0/BiasAdd/ReadVariableOpReadVariableOp1catdogmodel_conv0_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(CatDogModel/conv0/BiasAdd/ReadVariableOp�
CatDogModel/conv0/BiasAddBiasAdd!CatDogModel/conv0/Conv2D:output:00CatDogModel/conv0/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ 2
CatDogModel/conv0/BiasAdd�
CatDogModel/bn0/ReadVariableOpReadVariableOp'catdogmodel_bn0_readvariableop_resource*
_output_shapes
: *
dtype02 
CatDogModel/bn0/ReadVariableOp�
 CatDogModel/bn0/ReadVariableOp_1ReadVariableOp)catdogmodel_bn0_readvariableop_1_resource*
_output_shapes
: *
dtype02"
 CatDogModel/bn0/ReadVariableOp_1�
/CatDogModel/bn0/FusedBatchNormV3/ReadVariableOpReadVariableOp8catdogmodel_bn0_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype021
/CatDogModel/bn0/FusedBatchNormV3/ReadVariableOp�
1CatDogModel/bn0/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp:catdogmodel_bn0_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype023
1CatDogModel/bn0/FusedBatchNormV3/ReadVariableOp_1�
 CatDogModel/bn0/FusedBatchNormV3FusedBatchNormV3"CatDogModel/conv0/BiasAdd:output:0&CatDogModel/bn0/ReadVariableOp:value:0(CatDogModel/bn0/ReadVariableOp_1:value:07CatDogModel/bn0/FusedBatchNormV3/ReadVariableOp:value:09CatDogModel/bn0/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@@ : : : : :*
epsilon%o�:*
is_training( 2"
 CatDogModel/bn0/FusedBatchNormV3�
CatDogModel/activation/ReluRelu$CatDogModel/bn0/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������@@ 2
CatDogModel/activation/Relu�
CatDogModel/max_pool0/MaxPoolMaxPool)CatDogModel/activation/Relu:activations:0*/
_output_shapes
:���������   *
ksize
*
paddingVALID*
strides
2
CatDogModel/max_pool0/MaxPool�
CatDogModel/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� �  2
CatDogModel/flatten/Const�
CatDogModel/flatten/ReshapeReshape&CatDogModel/max_pool0/MaxPool:output:0"CatDogModel/flatten/Const:output:0*
T0*)
_output_shapes
:�����������2
CatDogModel/flatten/Reshape�
$CatDogModel/fc/MatMul/ReadVariableOpReadVariableOp-catdogmodel_fc_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02&
$CatDogModel/fc/MatMul/ReadVariableOp�
CatDogModel/fc/MatMulMatMul$CatDogModel/flatten/Reshape:output:0,CatDogModel/fc/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
CatDogModel/fc/MatMul�
%CatDogModel/fc/BiasAdd/ReadVariableOpReadVariableOp.catdogmodel_fc_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%CatDogModel/fc/BiasAdd/ReadVariableOp�
CatDogModel/fc/BiasAddBiasAddCatDogModel/fc/MatMul:product:0-CatDogModel/fc/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
CatDogModel/fc/BiasAdd�
CatDogModel/fc/SoftmaxSoftmaxCatDogModel/fc/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
CatDogModel/fc/Softmaxt
IdentityIdentity CatDogModel/fc/Softmax:softmax:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:���������@@:::::::::X T
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
: 
�$
�
=__inference_bn0_layer_call_and_return_conditional_losses_2018

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
*__inference_CatDogModel_layer_call_fn_1900

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*'
_output_shapes
:���������**
_read_only_resource_inputs

**
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_CatDogModel_layer_call_and_return_conditional_losses_17092
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:���������@@::::::::22
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
: 
�
y
$__inference_conv0_layer_call_fn_1316

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
GPU 

CPU2J 8*H
fCRA
?__inference_conv0_layer_call_and_return_conditional_losses_13062
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
�
D
(__inference_max_pool0_layer_call_fn_1454

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
GPU 

CPU2J 8*L
fGRE
C__inference_max_pool0_layer_call_and_return_conditional_losses_14482
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
�
�
=__inference_bn0_layer_call_and_return_conditional_losses_1431

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
=__inference_bn0_layer_call_and_return_conditional_losses_1961

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
7:���������@@ : : : : :*
epsilon%o�:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������@@ 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@@ :::::W S
/
_output_shapes
:���������@@ 
 
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
�~
�
 __inference__traced_restore_2316
file_prefix!
assignvariableop_conv0_kernel!
assignvariableop_1_conv0_bias 
assignvariableop_2_bn0_gamma
assignvariableop_3_bn0_beta&
"assignvariableop_4_bn0_moving_mean*
&assignvariableop_5_bn0_moving_variance 
assignvariableop_6_fc_kernel
assignvariableop_7_fc_bias 
assignvariableop_8_adam_iter"
assignvariableop_9_adam_beta_1#
assignvariableop_10_adam_beta_2"
assignvariableop_11_adam_decay*
&assignvariableop_12_adam_learning_rate
assignvariableop_13_total
assignvariableop_14_count
assignvariableop_15_total_1
assignvariableop_16_count_1+
'assignvariableop_17_adam_conv0_kernel_m)
%assignvariableop_18_adam_conv0_bias_m(
$assignvariableop_19_adam_bn0_gamma_m'
#assignvariableop_20_adam_bn0_beta_m(
$assignvariableop_21_adam_fc_kernel_m&
"assignvariableop_22_adam_fc_bias_m+
'assignvariableop_23_adam_conv0_kernel_v)
%assignvariableop_24_adam_conv0_bias_v(
$assignvariableop_25_adam_bn0_gamma_v'
#assignvariableop_26_adam_bn0_beta_v(
$assignvariableop_27_adam_fc_kernel_v&
"assignvariableop_28_adam_fc_bias_v
identity_30��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�	RestoreV2�RestoreV2_1�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*M
valueDBBB B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapesv
t:::::::::::::::::::::::::::::*+
dtypes!
2	2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOpassignvariableop_conv0_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv0_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOpassignvariableop_2_bn0_gammaIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOpassignvariableop_3_bn0_betaIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp"assignvariableop_4_bn0_moving_meanIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOp&assignvariableop_5_bn0_moving_varianceIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOpassignvariableop_6_fc_kernelIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOpassignvariableop_7_fc_biasIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0	*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_iterIdentity_8:output:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_1Identity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_2Identity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_decayIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp&assignvariableop_12_adam_learning_rateIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOpassignvariableop_15_total_1Identity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOpassignvariableop_16_count_1Identity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOp'assignvariableop_17_adam_conv0_kernel_mIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOp%assignvariableop_18_adam_conv0_bias_mIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOp$assignvariableop_19_adam_bn0_gamma_mIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOp#assignvariableop_20_adam_bn0_beta_mIdentity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOp$assignvariableop_21_adam_fc_kernel_mIdentity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOp"assignvariableop_22_adam_fc_bias_mIdentity_22:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_22_
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOp'assignvariableop_23_adam_conv0_kernel_vIdentity_23:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_23_
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOp%assignvariableop_24_adam_conv0_bias_vIdentity_24:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_24_
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOp$assignvariableop_25_adam_bn0_gamma_vIdentity_25:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_25_
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:2
Identity_26�
AssignVariableOp_26AssignVariableOp#assignvariableop_26_adam_bn0_beta_vIdentity_26:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_26_
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:2
Identity_27�
AssignVariableOp_27AssignVariableOp$assignvariableop_27_adam_fc_kernel_vIdentity_27:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_27_
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:2
Identity_28�
AssignVariableOp_28AssignVariableOp"assignvariableop_28_adam_fc_bias_vIdentity_28:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_28�
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
NoOp�
Identity_29Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_29�
Identity_30IdentityIdentity_29:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_30"#
identity_30Identity_30:output:0*�
_input_shapesx
v: :::::::::::::::::::::::::::::2$
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
AssignVariableOp_28AssignVariableOp_282(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
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
: 
�F
�
__inference__traced_save_2217
file_prefix+
'savev2_conv0_kernel_read_readvariableop)
%savev2_conv0_bias_read_readvariableop(
$savev2_bn0_gamma_read_readvariableop'
#savev2_bn0_beta_read_readvariableop.
*savev2_bn0_moving_mean_read_readvariableop2
.savev2_bn0_moving_variance_read_readvariableop(
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
"savev2_count_1_read_readvariableop2
.savev2_adam_conv0_kernel_m_read_readvariableop0
,savev2_adam_conv0_bias_m_read_readvariableop/
+savev2_adam_bn0_gamma_m_read_readvariableop.
*savev2_adam_bn0_beta_m_read_readvariableop/
+savev2_adam_fc_kernel_m_read_readvariableop-
)savev2_adam_fc_bias_m_read_readvariableop2
.savev2_adam_conv0_kernel_v_read_readvariableop0
,savev2_adam_conv0_bias_v_read_readvariableop/
+savev2_adam_bn0_gamma_v_read_readvariableop.
*savev2_adam_bn0_beta_v_read_readvariableop/
+savev2_adam_fc_kernel_v_read_readvariableop-
)savev2_adam_fc_bias_v_read_readvariableop
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
value3B1 B+_temp_66b4bb1a922a4da1b347ab047513c9c7/part2	
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
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*M
valueDBBB B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_conv0_kernel_read_readvariableop%savev2_conv0_bias_read_readvariableop$savev2_bn0_gamma_read_readvariableop#savev2_bn0_beta_read_readvariableop*savev2_bn0_moving_mean_read_readvariableop.savev2_bn0_moving_variance_read_readvariableop$savev2_fc_kernel_read_readvariableop"savev2_fc_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop.savev2_adam_conv0_kernel_m_read_readvariableop,savev2_adam_conv0_bias_m_read_readvariableop+savev2_adam_bn0_gamma_m_read_readvariableop*savev2_adam_bn0_beta_m_read_readvariableop+savev2_adam_fc_kernel_m_read_readvariableop)savev2_adam_fc_bias_m_read_readvariableop.savev2_adam_conv0_kernel_v_read_readvariableop,savev2_adam_conv0_bias_v_read_readvariableop+savev2_adam_bn0_gamma_v_read_readvariableop*savev2_adam_bn0_beta_v_read_readvariableop+savev2_adam_fc_kernel_v_read_readvariableop)savev2_adam_fc_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *+
dtypes!
2	2
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

identity_1Identity_1:output:0*�
_input_shapes�
�: : : : : : : :
��:: : : : : : : : : : : : : :
��:: : : : :
��:: 2(
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
: : 
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
: :&"
 
_output_shapes
:
��: 

_output_shapes
::	
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
: :,(
&
_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :&"
 
_output_shapes
:
��: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :&"
 
_output_shapes
:
��: 

_output_shapes
::

_output_shapes
: 
�	
�
*__inference_CatDogModel_layer_call_fn_1680
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*'
_output_shapes
:���������*(
_read_only_resource_inputs

**
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_CatDogModel_layer_call_and_return_conditional_losses_16612
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:���������@@::::::::22
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
: 
�
B
&__inference_flatten_layer_call_fn_2083

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*)
_output_shapes
:�����������* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_15682
PartitionedCalln
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������   :W S
/
_output_shapes
:���������   
 
_user_specified_nameinputs
�
]
A__inference_flatten_layer_call_and_return_conditional_losses_2078

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"���� �  2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:�����������2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������   :W S
/
_output_shapes
:���������   
 
_user_specified_nameinputs
�
�
"__inference_bn0_layer_call_fn_2049

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
GPU 

CPU2J 8*F
fAR?
=__inference_bn0_layer_call_and_return_conditional_losses_14002
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
=__inference_bn0_layer_call_and_return_conditional_losses_1494

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
7:���������@@ : : : : :*
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
:���������@@ 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@@ ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:���������@@ 
 
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
=__inference_bn0_layer_call_and_return_conditional_losses_1512

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
7:���������@@ : : : : :*
epsilon%o�:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������@@ 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@@ :::::W S
/
_output_shapes
:���������@@ 
 
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
"__inference_bn0_layer_call_fn_1987

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
:���������@@ *&
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*F
fAR?
=__inference_bn0_layer_call_and_return_conditional_losses_15122
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������@@ 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@@ ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������@@ 
 
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
"__inference_bn0_layer_call_fn_1974

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
:���������@@ *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*F
fAR?
=__inference_bn0_layer_call_and_return_conditional_losses_14942
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������@@ 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@@ ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������@@ 
 
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
�;
�
E__inference_CatDogModel_layer_call_and_return_conditional_losses_1821

inputs(
$conv0_conv2d_readvariableop_resource)
%conv0_biasadd_readvariableop_resource
bn0_readvariableop_resource!
bn0_readvariableop_1_resource0
,bn0_fusedbatchnormv3_readvariableop_resource2
.bn0_fusedbatchnormv3_readvariableop_1_resource%
!fc_matmul_readvariableop_resource&
"fc_biasadd_readvariableop_resource
identity��'bn0/AssignMovingAvg/AssignSubVariableOp�)bn0/AssignMovingAvg_1/AssignSubVariableOp�
zero_padding2d/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             2
zero_padding2d/Pad/paddings�
zero_padding2d/PadPadinputs$zero_padding2d/Pad/paddings:output:0*
T0*/
_output_shapes
:���������FF2
zero_padding2d/Pad�
conv0/Conv2D/ReadVariableOpReadVariableOp$conv0_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv0/Conv2D/ReadVariableOp�
conv0/Conv2DConv2Dzero_padding2d/Pad:output:0#conv0/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ *
paddingVALID*
strides
2
conv0/Conv2D�
conv0/BiasAdd/ReadVariableOpReadVariableOp%conv0_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv0/BiasAdd/ReadVariableOp�
conv0/BiasAddBiasAddconv0/Conv2D:output:0$conv0/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@ 2
conv0/BiasAdd�
bn0/ReadVariableOpReadVariableOpbn0_readvariableop_resource*
_output_shapes
: *
dtype02
bn0/ReadVariableOp�
bn0/ReadVariableOp_1ReadVariableOpbn0_readvariableop_1_resource*
_output_shapes
: *
dtype02
bn0/ReadVariableOp_1�
#bn0/FusedBatchNormV3/ReadVariableOpReadVariableOp,bn0_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02%
#bn0/FusedBatchNormV3/ReadVariableOp�
%bn0/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp.bn0_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02'
%bn0/FusedBatchNormV3/ReadVariableOp_1�
bn0/FusedBatchNormV3FusedBatchNormV3conv0/BiasAdd:output:0bn0/ReadVariableOp:value:0bn0/ReadVariableOp_1:value:0+bn0/FusedBatchNormV3/ReadVariableOp:value:0-bn0/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@@ : : : : :*
epsilon%o�:2
bn0/FusedBatchNormV3[
	bn0/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
	bn0/Const�
bn0/AssignMovingAvg/sub/xConst*?
_class5
31loc:@bn0/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2
bn0/AssignMovingAvg/sub/x�
bn0/AssignMovingAvg/subSub"bn0/AssignMovingAvg/sub/x:output:0bn0/Const:output:0*
T0*?
_class5
31loc:@bn0/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
bn0/AssignMovingAvg/sub�
"bn0/AssignMovingAvg/ReadVariableOpReadVariableOp,bn0_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02$
"bn0/AssignMovingAvg/ReadVariableOp�
bn0/AssignMovingAvg/sub_1Sub*bn0/AssignMovingAvg/ReadVariableOp:value:0!bn0/FusedBatchNormV3:batch_mean:0*
T0*?
_class5
31loc:@bn0/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
bn0/AssignMovingAvg/sub_1�
bn0/AssignMovingAvg/mulMulbn0/AssignMovingAvg/sub_1:z:0bn0/AssignMovingAvg/sub:z:0*
T0*?
_class5
31loc:@bn0/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
bn0/AssignMovingAvg/mul�
'bn0/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,bn0_fusedbatchnormv3_readvariableop_resourcebn0/AssignMovingAvg/mul:z:0#^bn0/AssignMovingAvg/ReadVariableOp$^bn0/FusedBatchNormV3/ReadVariableOp*?
_class5
31loc:@bn0/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02)
'bn0/AssignMovingAvg/AssignSubVariableOp�
bn0/AssignMovingAvg_1/sub/xConst*A
_class7
53loc:@bn0/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  �?2
bn0/AssignMovingAvg_1/sub/x�
bn0/AssignMovingAvg_1/subSub$bn0/AssignMovingAvg_1/sub/x:output:0bn0/Const:output:0*
T0*A
_class7
53loc:@bn0/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
bn0/AssignMovingAvg_1/sub�
$bn0/AssignMovingAvg_1/ReadVariableOpReadVariableOp.bn0_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02&
$bn0/AssignMovingAvg_1/ReadVariableOp�
bn0/AssignMovingAvg_1/sub_1Sub,bn0/AssignMovingAvg_1/ReadVariableOp:value:0%bn0/FusedBatchNormV3:batch_variance:0*
T0*A
_class7
53loc:@bn0/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
bn0/AssignMovingAvg_1/sub_1�
bn0/AssignMovingAvg_1/mulMulbn0/AssignMovingAvg_1/sub_1:z:0bn0/AssignMovingAvg_1/sub:z:0*
T0*A
_class7
53loc:@bn0/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
bn0/AssignMovingAvg_1/mul�
)bn0/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.bn0_fusedbatchnormv3_readvariableop_1_resourcebn0/AssignMovingAvg_1/mul:z:0%^bn0/AssignMovingAvg_1/ReadVariableOp&^bn0/FusedBatchNormV3/ReadVariableOp_1*A
_class7
53loc:@bn0/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02+
)bn0/AssignMovingAvg_1/AssignSubVariableOp~
activation/ReluRelubn0/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������@@ 2
activation/Relu�
max_pool0/MaxPoolMaxPoolactivation/Relu:activations:0*/
_output_shapes
:���������   *
ksize
*
paddingVALID*
strides
2
max_pool0/MaxPoolo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� �  2
flatten/Const�
flatten/ReshapeReshapemax_pool0/MaxPool:output:0flatten/Const:output:0*
T0*)
_output_shapes
:�����������2
flatten/Reshape�
fc/MatMul/ReadVariableOpReadVariableOp!fc_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
fc/MatMul/ReadVariableOp�
	fc/MatMulMatMulflatten/Reshape:output:0 fc/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
	fc/MatMul�
fc/BiasAdd/ReadVariableOpReadVariableOp"fc_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
fc/BiasAdd/ReadVariableOp�

fc/BiasAddBiasAddfc/MatMul:product:0!fc/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2

fc/BiasAddj

fc/SoftmaxSoftmaxfc/BiasAdd:output:0*
T0*'
_output_shapes
:���������2

fc/Softmax�
IdentityIdentityfc/Softmax:softmax:0(^bn0/AssignMovingAvg/AssignSubVariableOp*^bn0/AssignMovingAvg_1/AssignSubVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:���������@@::::::::2R
'bn0/AssignMovingAvg/AssignSubVariableOp'bn0/AssignMovingAvg/AssignSubVariableOp2V
)bn0/AssignMovingAvg_1/AssignSubVariableOp)bn0/AssignMovingAvg_1/AssignSubVariableOp:W S
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
: 
�
�
<__inference_fc_layer_call_and_return_conditional_losses_1587

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
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
identityIdentity:output:0*0
_input_shapes
:�����������:::Q M
)
_output_shapes
:�����������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�	
�
*__inference_CatDogModel_layer_call_fn_1879

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*'
_output_shapes
:���������*(
_read_only_resource_inputs

**
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_CatDogModel_layer_call_and_return_conditional_losses_16612
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:���������@@::::::::22
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
: 
�
I
-__inference_zero_padding2d_layer_call_fn_1295

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
GPU 

CPU2J 8*Q
fLRJ
H__inference_zero_padding2d_layer_call_and_return_conditional_losses_12892
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
�$
�
=__inference_bn0_layer_call_and_return_conditional_losses_1400

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
�
v
!__inference_fc_layer_call_fn_2103

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
GPU 

CPU2J 8*E
f@R>
<__inference_fc_layer_call_and_return_conditional_losses_15872
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*0
_input_shapes
:�����������::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:�����������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
�
E__inference_CatDogModel_layer_call_and_return_conditional_losses_1661

inputs

conv0_1638

conv0_1640
bn0_1643
bn0_1645
bn0_1647
bn0_1649
fc_1655
fc_1657
identity��bn0/StatefulPartitionedCall�conv0/StatefulPartitionedCall�fc/StatefulPartitionedCall�
zero_padding2d/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*/
_output_shapes
:���������FF* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*Q
fLRJ
H__inference_zero_padding2d_layer_call_and_return_conditional_losses_12892 
zero_padding2d/PartitionedCall�
conv0/StatefulPartitionedCallStatefulPartitionedCall'zero_padding2d/PartitionedCall:output:0
conv0_1638
conv0_1640*
Tin
2*
Tout
2*/
_output_shapes
:���������@@ *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*H
fCRA
?__inference_conv0_layer_call_and_return_conditional_losses_13062
conv0/StatefulPartitionedCall�
bn0/StatefulPartitionedCallStatefulPartitionedCall&conv0/StatefulPartitionedCall:output:0bn0_1643bn0_1645bn0_1647bn0_1649*
Tin	
2*
Tout
2*/
_output_shapes
:���������@@ *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*F
fAR?
=__inference_bn0_layer_call_and_return_conditional_losses_14942
bn0/StatefulPartitionedCall�
activation/PartitionedCallPartitionedCall$bn0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:���������@@ * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_activation_layer_call_and_return_conditional_losses_15532
activation/PartitionedCall�
max_pool0/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:���������   * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_max_pool0_layer_call_and_return_conditional_losses_14482
max_pool0/PartitionedCall�
flatten/PartitionedCallPartitionedCall"max_pool0/PartitionedCall:output:0*
Tin
2*
Tout
2*)
_output_shapes
:�����������* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_15682
flatten/PartitionedCall�
fc/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0fc_1655fc_1657*
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
GPU 

CPU2J 8*E
f@R>
<__inference_fc_layer_call_and_return_conditional_losses_15872
fc/StatefulPartitionedCall�
IdentityIdentity#fc/StatefulPartitionedCall:output:0^bn0/StatefulPartitionedCall^conv0/StatefulPartitionedCall^fc/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:���������@@::::::::2:
bn0/StatefulPartitionedCallbn0/StatefulPartitionedCall2>
conv0/StatefulPartitionedCallconv0/StatefulPartitionedCall28
fc/StatefulPartitionedCallfc/StatefulPartitionedCall:W S
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
: 
�
�
<__inference_fc_layer_call_and_return_conditional_losses_2094

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
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
identityIdentity:output:0*0
_input_shapes
:�����������:::Q M
)
_output_shapes
:�����������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
�
E__inference_CatDogModel_layer_call_and_return_conditional_losses_1604
input_1

conv0_1459

conv0_1461
bn0_1539
bn0_1541
bn0_1543
bn0_1545
fc_1598
fc_1600
identity��bn0/StatefulPartitionedCall�conv0/StatefulPartitionedCall�fc/StatefulPartitionedCall�
zero_padding2d/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*/
_output_shapes
:���������FF* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*Q
fLRJ
H__inference_zero_padding2d_layer_call_and_return_conditional_losses_12892 
zero_padding2d/PartitionedCall�
conv0/StatefulPartitionedCallStatefulPartitionedCall'zero_padding2d/PartitionedCall:output:0
conv0_1459
conv0_1461*
Tin
2*
Tout
2*/
_output_shapes
:���������@@ *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*H
fCRA
?__inference_conv0_layer_call_and_return_conditional_losses_13062
conv0/StatefulPartitionedCall�
bn0/StatefulPartitionedCallStatefulPartitionedCall&conv0/StatefulPartitionedCall:output:0bn0_1539bn0_1541bn0_1543bn0_1545*
Tin	
2*
Tout
2*/
_output_shapes
:���������@@ *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*F
fAR?
=__inference_bn0_layer_call_and_return_conditional_losses_14942
bn0/StatefulPartitionedCall�
activation/PartitionedCallPartitionedCall$bn0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:���������@@ * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_activation_layer_call_and_return_conditional_losses_15532
activation/PartitionedCall�
max_pool0/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:���������   * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_max_pool0_layer_call_and_return_conditional_losses_14482
max_pool0/PartitionedCall�
flatten/PartitionedCallPartitionedCall"max_pool0/PartitionedCall:output:0*
Tin
2*
Tout
2*)
_output_shapes
:�����������* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_15682
flatten/PartitionedCall�
fc/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0fc_1598fc_1600*
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
GPU 

CPU2J 8*E
f@R>
<__inference_fc_layer_call_and_return_conditional_losses_15872
fc/StatefulPartitionedCall�
IdentityIdentity#fc/StatefulPartitionedCall:output:0^bn0/StatefulPartitionedCall^conv0/StatefulPartitionedCall^fc/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:���������@@::::::::2:
bn0/StatefulPartitionedCallbn0/StatefulPartitionedCall2>
conv0/StatefulPartitionedCallconv0/StatefulPartitionedCall28
fc/StatefulPartitionedCallfc/StatefulPartitionedCall:X T
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
: 
�
�
=__inference_bn0_layer_call_and_return_conditional_losses_2036

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
�
E
)__inference_activation_layer_call_fn_2072

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*/
_output_shapes
:���������@@ * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_activation_layer_call_and_return_conditional_losses_15532
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������@@ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@@ :W S
/
_output_shapes
:���������@@ 
 
_user_specified_nameinputs
�

�
*__inference_CatDogModel_layer_call_fn_1728
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*'
_output_shapes
:���������**
_read_only_resource_inputs

**
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_CatDogModel_layer_call_and_return_conditional_losses_17092
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:���������@@::::::::22
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
serving_default_input_1:0���������@@6
fc0
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�;
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer-6
layer_with_weights-2
layer-7
		optimizer

	variables
trainable_variables
regularization_losses
	keras_api

signatures
x_default_save_signature
y__call__
*z&call_and_return_all_conditional_losses"�8
_tf_keras_model�8{"class_name": "Model", "name": "CatDogModel", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "CatDogModel", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "ZeroPadding2D", "config": {"name": "zero_padding2d", "trainable": true, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [3, 3]}, {"class_name": "__tuple__", "items": [3, 3]}]}, "data_format": "channels_last"}, "name": "zero_padding2d", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv0", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [7, 7]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv0", "inbound_nodes": [[["zero_padding2d", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "bn0", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "bn0", "inbound_nodes": [[["conv0", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation", "inbound_nodes": [[["bn0", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pool0", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pool0", "inbound_nodes": [[["activation", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["max_pool0", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "fc", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "fc", "inbound_nodes": [[["flatten", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["fc", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 3]}, "is_graph_network": true, "keras_version": "2.3.0-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "CatDogModel", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "ZeroPadding2D", "config": {"name": "zero_padding2d", "trainable": true, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [3, 3]}, {"class_name": "__tuple__", "items": [3, 3]}]}, "data_format": "channels_last"}, "name": "zero_padding2d", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv0", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [7, 7]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv0", "inbound_nodes": [[["zero_padding2d", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "bn0", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "bn0", "inbound_nodes": [[["conv0", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation", "inbound_nodes": [[["bn0", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pool0", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pool0", "inbound_nodes": [[["activation", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["max_pool0", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "fc", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "fc", "inbound_nodes": [[["flatten", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["fc", 0, 0]]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": ["accuracy"], "weighted_metrics": null, "loss_weights": null, "sample_weight_mode": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
�
	variables
trainable_variables
regularization_losses
	keras_api
{__call__
*|&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "ZeroPadding2D", "name": "zero_padding2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "zero_padding2d", "trainable": true, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [3, 3]}, {"class_name": "__tuple__", "items": [3, 3]}]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�	

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
}__call__
*~&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv0", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [7, 7]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 70, 70, 3]}}
�
axis
	gamma
beta
moving_mean
moving_variance
	variables
trainable_variables
 regularization_losses
!	keras_api
__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "bn0", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "bn0", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 32]}}
�
"	variables
#trainable_variables
$regularization_losses
%	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Activation", "name": "activation", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}}
�
&	variables
'trainable_variables
(regularization_losses
)	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "MaxPooling2D", "name": "max_pool0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "max_pool0", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�
*	variables
+trainable_variables
,regularization_losses
-	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
�

.kernel
/bias
0	variables
1trainable_variables
2regularization_losses
3	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "fc", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "fc", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32768}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32768]}}
�
4iter

5beta_1

6beta_2
	7decay
8learning_ratemlmmmnmo.mp/mqvrvsvtvu.vv/vw"
	optimizer
X
0
1
2
3
4
5
.6
/7"
trackable_list_wrapper
J
0
1
2
3
.4
/5"
trackable_list_wrapper
 "
trackable_list_wrapper
�
9layer_metrics
:non_trainable_variables

;layers
<layer_regularization_losses

	variables
trainable_variables
=metrics
regularization_losses
y__call__
x_default_save_signature
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
>layer_metrics
?non_trainable_variables

@layers
Alayer_regularization_losses
	variables
trainable_variables
Bmetrics
regularization_losses
{__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses"
_generic_user_object
&:$ 2conv0/kernel
: 2
conv0/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Clayer_metrics
Dnon_trainable_variables

Elayers
Flayer_regularization_losses
	variables
trainable_variables
Gmetrics
regularization_losses
}__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
: 2	bn0/gamma
: 2bn0/beta
:  (2bn0/moving_mean
#:!  (2bn0/moving_variance
<
0
1
2
3"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Hlayer_metrics
Inon_trainable_variables

Jlayers
Klayer_regularization_losses
	variables
trainable_variables
Lmetrics
 regularization_losses
__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
Mlayer_metrics
Nnon_trainable_variables

Olayers
Player_regularization_losses
"	variables
#trainable_variables
Qmetrics
$regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
Rlayer_metrics
Snon_trainable_variables

Tlayers
Ulayer_regularization_losses
&	variables
'trainable_variables
Vmetrics
(regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
Wlayer_metrics
Xnon_trainable_variables

Ylayers
Zlayer_regularization_losses
*	variables
+trainable_variables
[metrics
,regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
:
��2	fc/kernel
:2fc/bias
.
.0
/1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
\layer_metrics
]non_trainable_variables

^layers
_layer_regularization_losses
0	variables
1trainable_variables
`metrics
2regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
.
a0
b1"
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
.
0
1"
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
�
	ctotal
	dcount
e	variables
f	keras_api"�
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
�
	gtotal
	hcount
i
_fn_kwargs
j	variables
k	keras_api"�
_tf_keras_metric�{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}
:  (2total
:  (2count
.
c0
d1"
trackable_list_wrapper
-
e	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
g0
h1"
trackable_list_wrapper
-
j	variables"
_generic_user_object
+:) 2Adam/conv0/kernel/m
: 2Adam/conv0/bias/m
: 2Adam/bn0/gamma/m
: 2Adam/bn0/beta/m
": 
��2Adam/fc/kernel/m
:2Adam/fc/bias/m
+:) 2Adam/conv0/kernel/v
: 2Adam/conv0/bias/v
: 2Adam/bn0/gamma/v
: 2Adam/bn0/beta/v
": 
��2Adam/fc/kernel/v
:2Adam/fc/bias/v
�2�
__inference__wrapped_model_1282�
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
�2�
*__inference_CatDogModel_layer_call_fn_1680
*__inference_CatDogModel_layer_call_fn_1900
*__inference_CatDogModel_layer_call_fn_1879
*__inference_CatDogModel_layer_call_fn_1728�
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
�2�
E__inference_CatDogModel_layer_call_and_return_conditional_losses_1821
E__inference_CatDogModel_layer_call_and_return_conditional_losses_1631
E__inference_CatDogModel_layer_call_and_return_conditional_losses_1604
E__inference_CatDogModel_layer_call_and_return_conditional_losses_1858�
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
-__inference_zero_padding2d_layer_call_fn_1295�
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
H__inference_zero_padding2d_layer_call_and_return_conditional_losses_1289�
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
$__inference_conv0_layer_call_fn_1316�
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
?__inference_conv0_layer_call_and_return_conditional_losses_1306�
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
"__inference_bn0_layer_call_fn_1987
"__inference_bn0_layer_call_fn_2062
"__inference_bn0_layer_call_fn_2049
"__inference_bn0_layer_call_fn_1974�
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
�2�
=__inference_bn0_layer_call_and_return_conditional_losses_1961
=__inference_bn0_layer_call_and_return_conditional_losses_2036
=__inference_bn0_layer_call_and_return_conditional_losses_2018
=__inference_bn0_layer_call_and_return_conditional_losses_1943�
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
)__inference_activation_layer_call_fn_2072�
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
D__inference_activation_layer_call_and_return_conditional_losses_2067�
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
(__inference_max_pool0_layer_call_fn_1454�
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
C__inference_max_pool0_layer_call_and_return_conditional_losses_1448�
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
�2�
&__inference_flatten_layer_call_fn_2083�
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
A__inference_flatten_layer_call_and_return_conditional_losses_2078�
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
!__inference_fc_layer_call_fn_2103�
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
<__inference_fc_layer_call_and_return_conditional_losses_2094�
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
1B/
"__inference_signature_wrapper_1771input_1�
E__inference_CatDogModel_layer_call_and_return_conditional_losses_1604s./@�=
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
E__inference_CatDogModel_layer_call_and_return_conditional_losses_1631s./@�=
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
E__inference_CatDogModel_layer_call_and_return_conditional_losses_1821r./?�<
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
E__inference_CatDogModel_layer_call_and_return_conditional_losses_1858r./?�<
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
*__inference_CatDogModel_layer_call_fn_1680f./@�=
6�3
)�&
input_1���������@@
p

 
� "�����������
*__inference_CatDogModel_layer_call_fn_1728f./@�=
6�3
)�&
input_1���������@@
p 

 
� "�����������
*__inference_CatDogModel_layer_call_fn_1879e./?�<
5�2
(�%
inputs���������@@
p

 
� "�����������
*__inference_CatDogModel_layer_call_fn_1900e./?�<
5�2
(�%
inputs���������@@
p 

 
� "�����������
__inference__wrapped_model_1282m./8�5
.�+
)�&
input_1���������@@
� "'�$
"
fc�
fc����������
D__inference_activation_layer_call_and_return_conditional_losses_2067h7�4
-�*
(�%
inputs���������@@ 
� "-�*
#� 
0���������@@ 
� �
)__inference_activation_layer_call_fn_2072[7�4
-�*
(�%
inputs���������@@ 
� " ����������@@ �
=__inference_bn0_layer_call_and_return_conditional_losses_1943r;�8
1�.
(�%
inputs���������@@ 
p
� "-�*
#� 
0���������@@ 
� �
=__inference_bn0_layer_call_and_return_conditional_losses_1961r;�8
1�.
(�%
inputs���������@@ 
p 
� "-�*
#� 
0���������@@ 
� �
=__inference_bn0_layer_call_and_return_conditional_losses_2018�M�J
C�@
:�7
inputs+��������������������������� 
p
� "?�<
5�2
0+��������������������������� 
� �
=__inference_bn0_layer_call_and_return_conditional_losses_2036�M�J
C�@
:�7
inputs+��������������������������� 
p 
� "?�<
5�2
0+��������������������������� 
� �
"__inference_bn0_layer_call_fn_1974e;�8
1�.
(�%
inputs���������@@ 
p
� " ����������@@ �
"__inference_bn0_layer_call_fn_1987e;�8
1�.
(�%
inputs���������@@ 
p 
� " ����������@@ �
"__inference_bn0_layer_call_fn_2049�M�J
C�@
:�7
inputs+��������������������������� 
p
� "2�/+��������������������������� �
"__inference_bn0_layer_call_fn_2062�M�J
C�@
:�7
inputs+��������������������������� 
p 
� "2�/+��������������������������� �
?__inference_conv0_layer_call_and_return_conditional_losses_1306�I�F
?�<
:�7
inputs+���������������������������
� "?�<
5�2
0+��������������������������� 
� �
$__inference_conv0_layer_call_fn_1316�I�F
?�<
:�7
inputs+���������������������������
� "2�/+��������������������������� �
<__inference_fc_layer_call_and_return_conditional_losses_2094^./1�.
'�$
"�
inputs�����������
� "%�"
�
0���������
� v
!__inference_fc_layer_call_fn_2103Q./1�.
'�$
"�
inputs�����������
� "�����������
A__inference_flatten_layer_call_and_return_conditional_losses_2078b7�4
-�*
(�%
inputs���������   
� "'�$
�
0�����������
� 
&__inference_flatten_layer_call_fn_2083U7�4
-�*
(�%
inputs���������   
� "�������������
C__inference_max_pool0_layer_call_and_return_conditional_losses_1448�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
(__inference_max_pool0_layer_call_fn_1454�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
"__inference_signature_wrapper_1771x./C�@
� 
9�6
4
input_1)�&
input_1���������@@"'�$
"
fc�
fc����������
H__inference_zero_padding2d_layer_call_and_return_conditional_losses_1289�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
-__inference_zero_padding2d_layer_call_fn_1295�R�O
H�E
C�@
inputs4������������������������������������
� ";�84������������������������������������