#! /bin/sh
brew install rename
rm images/*.JPG
cd images/apple
rename -vs IMG_ apple_ *
find . -name '*.JPG' -exec ln {} ../{} \;
cd ../banana
rename -vs IMG_ banana_ *
find . -name '*.JPG' -exec ln {} ../{} \;
cd ../biscuit
rename -vs IMG_ biscuit_ *
find . -name '*.JPG' -exec ln {} ../{} \;
cd ../carrot
rename -vs IMG_ carrot_ *
find . -name '*.JPG' -exec ln {} ../{} \;
cd ../champignon
rename -vs IMG_ champignon_ *
find . -name '*.JPG' -exec ln {} ../{} \;
cd ../cheese
rename -vs IMG_ cheese_ *
find . -name '*.JPG' -exec ln {} ../{} \;
cd ../chocolate
rename -vs IMG_ chocolate_ *
find . -name '*.JPG' -exec ln {} ../{} \;
cd ../cow
rename -vs IMG_ cow_ *
find . -name '*.JPG' -exec ln {} ../{} \;
cd ../duck
rename -vs IMG_ duck_ *
find . -name '*.JPG' -exec ln {} ../{} \;
cd ../elephant
rename -vs IMG_ elephant_ *
find . -name '*.JPG' -exec ln {} ../{} \;
cd ../fish
rename -vs IMG_ fish_ *
find . -name '*.JPG' -exec ln {} ../{} \;
cd ../flamingo
rename -vs IMG_ flamingo_ *
find . -name '*.JPG' -exec ln {} ../{} \;
cd ../horse
rename -vs IMG_ horse_ *
find . -name '*.JPG' -exec ln {} ../{} \;
cd ../sponge-cake
rename -vs IMG_ sponge-cake_ *
find . -name '*.JPG' -exec ln {} ../{} \;

# ln -s images/apple/{} images/{} +;
