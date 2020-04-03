# Neural_Network_mk2
This is a follow-up to Neural\_Network\_mk1 written in C++ (although it really
could have been written in C; we went very low-level for this project). This is
a simple neural network written from scratch in C++ by myself and Alex Xuan
(silvernx on Github). 

## General info
This is a massive improvement on Neural\_Network\_mk1 in every way except maybe
interface. Like Neural\_Network\_mk1 there are three versions here: a
singlethreaded version, a multithreaded version, and a version learning the
MNIST dataset. It is orders of magnitude faster than than Neural\_Network\_mk1.
Everything is implemented as raw C-style arrays with optimizations everywhere.
The multithreading is done in `pthreads` (and requires barriers, so
unfortunately it does not work on OSX). 

## How to use
I sincerely apologize for the incredibly unfriendly `driver.cpp`. You can read
through that file and see what you want to change (look for `epochs`,
`batch_size`, etc. as those are the interesting ones to tweak) then give it a
`make ; make run` or `make ; time make run` if you want to compare it to
Neural\_Network\_mk1 or marvel at how fast it is.
