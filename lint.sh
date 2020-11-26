dir =$1

if [-z "$dir"]
then
echo "\nUsage: sh lint.sh [directory]"
exit 0
fi

for file in $dir/**
do
autopep8 - i "$file"
done
