# 说明

## public 文件夹

public 文件夹有各种关于模型的定义已经模型的构建，还有模型保存方法的封装

## frame-base

为单帧通过 CNN 提取特征之后直接通过全连接层来进行回归

## time-base

为以 5 帧为单元，每一帧通过 VGG16 或者其他 CNN 提取特征之后 feed 进入 LSTM 中，初始状态为 0
