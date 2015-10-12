#!/usr/bin/perl -w

$num_jobs = $ARGV[0]; shift;
$base_filename = $ARGV[0]; shift;

@num_frames = (0) x $num_jobs;

foreach $i (1..$num_jobs) {
  local *FILE;
  open(FILE, "> $base_filename.$i.scp") || die;
  push(@file_handles, *FILE);
}

while (<>) {
  chomp;
  @A = split /\s+/;
  $id_min = 0;
  $num_frames[$id_min] < $num_frames[$_] or $id_min = $_ for 1 .. $#num_frames;    # find the smallest index
  print {$file_handles[$id_min]} $A[0],"\n";
  $num_frames[$id_min] += $A[1];
}

$id_min = 0;
$num_frames[$id_min] < $num_frames[$_] or $id_min = $_ for 1 .. $#num_frames;    # find the smallest index
print "$num_frames[$id_min]";
