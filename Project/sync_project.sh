#!/bin/sh

if [ $1 = "--remote" ]; then
	echo "Tranferring local data to remote."
	rsync -aP ./ hlcv_team021@conduit.cs.uni-saarland.de:/home/hlcv_team021/hlcv2021/Project
elif [ $1 = "--local" ]; then
	echo "Tranferring remote data to local."
	rsync -aP hlcv_team021@conduit.cs.uni-saarland.de:/home/hlcv_team021/hlcv2021/Project/ .
else
	echo "Provide a valid argument."
fi
