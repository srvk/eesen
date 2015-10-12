#!/usr/bin/perl -w

# Copyright 2015  Hang Su.   Apache 2.0.

# This script split an scp list either by length of the frames or in round-robin manner

$mode = 'frame';
if ($ARGV[0] eq '--mode') {
  shift @ARGV;
  $mode = shift @ARGV;
}

$num_jobs = $ARGV[0]; shift;
$base_filename = $ARGV[0]; shift;

@num_frames = (0) x $num_jobs;

foreach $i (1..$num_jobs) {
  local *FILE;
  open(FILE, "> $base_filename.$i.scp") || die;
  push(@file_handles, *FILE);
}

$count = 0;
while (<>) {
  chomp;
  if ($mode eq "utt") {
    $id = ($count % $num_jobs) ;
    print {$file_handles[$id]} $_,"\n";
  } elsif ($mode eq "frame") {
    @A = split /\s+/;
    $id_min = 0;
    $num_frames[$id_min] < $num_frames[$_] or $id_min = $_ for 1 .. $#num_frames;    # find the smallest index
    $id = $id_min;
    $num_frames[$id_min] += $A[1];
    print {$file_handles[$id]} $A[0],"\n";
  } else {
    die "Un-recognized mode $mode!";
  }
  $count += 1;
}

$id_min = 0;
$num_frames[$id_min] < $num_frames[$_] or $id_min = $_ for 1 .. $#num_frames;    # find the smallest index
print "$num_frames[$id_min]";
