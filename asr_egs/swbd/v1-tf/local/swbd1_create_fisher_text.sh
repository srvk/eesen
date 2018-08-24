



help_message="Usage: $0 <dest-dir> <fisher_a> <fisher_b> 
create cleaned (according to units.txt) with uttid text file from fisher data:
options:
  --help          # print this message and exit
  --weblm DIR     # directory for web-data from University of Washington
";


dir=$1
fsh_a=$2
fsh_b=$3

if [ $# -ne 3 ]; then
  printf "$help_message\n";
  exit 1;
fi

rm -r $dir

mkdir -p $dir


echo processing $fsh_a ...

if [ -d $fsh_a/data/trans ]; then
  cat $fsh_a/data/trans/*/*.txt |  grep -v ^# | grep -v ^$ | cut -d' ' -f4- >> $dir/text0
elif [ -d $fsh_a/DATA/TRANS ]; then
  cat $fsh_a/DATA/TRANS/*/*.TXT | grep -v ^# | grep -v ^$ | cut -d' ' -f4- >> $dir/text0
else
  echo "$0: Cannot find transcripts in Fisher directory $x" && exit 1;
fi

echo processing $fsh_b ...

if [ -d $fsh_b/data/trans ]; then
    cat $fsh_b/data/trans/*/*.txt |  grep -v ^# | grep -v ^$ | cut -d' ' -f4- >> $dir/text0
elif [ -d $fsh_b/DATA/TRANS ]; then
  cat $fsh_b/DATA/TRANS/*/*.TXT | grep -v ^# | grep -v ^$ | cut -d' ' -f4- >> $dir/text0
else
  echo "$0: Cannot find transcripts in Fisher directory $x" && exit 1;
fi



echo "adding fake utt_id..."
cat $dir/text0 | sed -e 's/^/feak_utt /g' > $dir/text
rm $dir/text0







