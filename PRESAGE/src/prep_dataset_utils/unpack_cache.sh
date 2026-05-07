tar -xvzf cache.tar.gz
mv ./cache/data/ .
mv ./cache/splits/ .
mv ./cache/configs/ .
find . | grep json.gz > temp.txt
while read line; do gunzip ${line}; done < temp.txt
