#! /bin/sh
cd images/apple
find . -name '*.JPG' -exec ln {} ../{} \;
cd ../banana
find . -name '*.JPG' -exec ln {} ../{} \;
cd ../carrot
find . -name '*.JPG' -exec ln {} ../{} \;
cd ../champignon
find . -name '*.JPG' -exec ln {} ../{} \;
cd ../cheese
find . -name '*.JPG' -exec ln {} ../{} \;
cd ../chocolate
find . -name '*.JPG' -exec ln {} ../{} \;
cd ../cow
find . -name '*.JPG' -exec ln {} ../{} \;
cd ../duck
find . -name '*.JPG' -exec ln {} ../{} \;
cd ../elephant
find . -name '*.JPG' -exec ln {} ../{} \;
cd ../fish
find . -name '*.JPG' -exec ln {} ../{} \;
cd ../flamingo
find . -name '*.JPG' -exec ln {} ../{} \;

# ln -s images/apple/{} images/{} +;
