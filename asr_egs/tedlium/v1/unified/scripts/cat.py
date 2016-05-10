import linecache
import glob
import sys

# Parse and append lines of praat_file and snr_file 
# in the style of the paste shell command


if len(sys.argv[1:]) < 2:
  print >>sys.stderr, "Usage: %s praat_result snr_result final.res" % sys.argv[0]
  sys.exit(1)

praat_file = sys.argv[1]
snr_file = sys.argv[2]
result = sys.argv[3]

para1 = open (praat_file, 'r')
para2 = open (snr_file, 'r')
para_result = open(result, 'w')
count_line = len(open(praat_file).readlines()) - 1

print "count_line: ", count_line

i = 1
while (i <= count_line):
	str1 = str(linecache.getline(praat_file, i)).rstrip()
	if str1[0:1]==" ":
		str1 = str1[1:]
	print str1
	str2 = str(linecache.getline(snr_file, i)).strip()
	tokens = str2.split(" : ")
	snr = tokens[-1]
	print snr
	t = i + 1 
	tmp = str1 + " " + snr 
	print >> para_result,  "%s" %(tmp)
	i = i + 1
