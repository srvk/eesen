#!/usr/bin/env python -O
# 
# Copyright (c) 2011-2013 Kyle Gorman, Max Bane, Morgan Sonderegger
# 
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
# 
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# textgrid.py: classes for Praat TextGrid and HTK mlf files
# 
# Max Bane <bane@uchicago.edu>
# Kyle Gorman <gormanky@ohsu.edu>
# Morgan Sonderegger <morgan.sonderegger@mcgill.ca>

###############################################################################
#
# Modification by yunwang on 10/18/2014:
#   Interval.__cmp__() no longer raises ValueError, so it is possible to read
#   TextGrid files with slightly overlapping intervals 
#
###############################################################################

import re
import codecs
import os.path

from sys import stderr
from bisect import bisect_left


def readFile(f):
    """
    This helper method returns an appropriate file handle given a path f.
    This handles UTF-8, which is itself an ASCII extension, so also ASCII.
    """
    return codecs.open(f, 'r', encoding='UTF-8')


class Point(object):
    """ 
    Represents a point in time with an associated textual mark, as stored 
    in a PointTier.

    # Point/Point comparison
    >>> foo = Point(3.0, 'foo')
    >>> bar = Point(4.0, 'bar')
    >>> foo < bar
    True
    >>> foo == Point(3.0, 'baz')
    True
    >>> bar > foo
    True

    # Point/Value comparison
    >>> foo < 4.0
    True
    >>> foo == 3.0
    True
    >>> foo > 5.0
    False

    # Point/Interval comparison
    >>> baz = Interval(3.0, 5.0, 'baz')
    >>> foo < baz
    True
    >>> foo == baz
    False
    >>> bar == baz 
    True
    """

    def __init__(self, time, mark):
        self.time = time
        self.mark = mark

    def __repr__(self):
        return 'Point({0}, {1})'.format(self.time, 
                                        self.mark if self.mark else None)

    def __cmp__(self, other):
        """
        In addition to the obvious semantics, Point/Interval comparison is
        0 iff the point is inside the interval (non-inclusively), if you 
        need inclusive membership, use Interval.__contains__
        """
        if hasattr(other, 'time'):
            return cmp(self.time, other.time)
        elif hasattr(other, 'minTime') and hasattr(other, 'maxTime'):
            return cmp(self.time, other.minTime) + \
                   cmp(self.time, other.maxTime)
        else: # hopefully numerical
            return cmp(self.time, other)

    def __iadd__(self, other):
        self.time += other

    def __isub__(self, other):
        self.time -= other


def decode(string):
    """
    Decode HTK's mangling of UTF-8 strings into something useful
    """
    return string.decode('string_escape').decode('UTF-8')

class Interval(object):
    """ 
    Represents an interval of time, with an associated textual mark, as 
    stored in an IntervalTier.

    >>> foo = Point(3.0, 'foo')
    >>> bar = Point(4.0, 'bar')
    >>> baz = Interval(3.0, 5.0, 'baz')
    >>> foo in baz
    True
    >>> 3.0 in baz
    True
    >>> bar in baz
    True
    >>> 4.0 in baz
    True
    """

    def __init__(self, minTime, maxTime, mark):
        if minTime > maxTime: # not an actual interval
            raise ValueError(minTime, maxTime)
        self.minTime = minTime
        self.maxTime = maxTime
        self.mark = mark

    def __repr__(self):
        return 'Interval({0}, {1}, {2})'.format(self.minTime, self.maxTime,
                                         self.mark if self.mark else None)

    def duration(self):
        """ 
        Returns the duration of the interval in seconds.
        """
        return self.maxTime - self.minTime

    def __cmp__(self, other):
        if hasattr(other, 'minTime') and hasattr(other, 'maxTime'):
# yunwang 10/18/2014: commented the next two lines out
#            if self.overlaps(other): 
#                raise ValueError(self, other)
                # this returns the two intervals, so user can patch things
                # up if s/he so chooses
            return cmp(self.minTime, other.minTime)
        elif hasattr(other, 'time'): # comparing Intervals and Points
            return cmp(self.minTime, other.time) + \
                   cmp(self.maxTime, other.time)
        else: 
            return cmp(self.minTime, other) + cmp(self.maxTime, other)

    def __eq__(self, other):
        """
        This might seem superfluous but not that a ValueError will be 
        raised if you compare two intervals to each other...not anymore
        """
        if hasattr(other, 'minTime') and hasattr(other, 'maxTime'):
            if self.minTime == other.minTime:
                if self.maxTime == other.maxTime:
                    return True
        elif hasattr(other, 'time'):
            return self.minTime < other.time < self.maxTime
        else:
            return False

    def __iadd__(self, other):
        self.minTime += other
        self.maxTime += other

    def __isub__(self, other):
        self.minTime -= other
        self.maxTime -= other

    def overlaps(self, other):
        """
        Tests whether self overlaps with the given interval. Symmetric.
        See: http://www.rgrjr.com/emacs/overlap.html
        """
        return other.minTime < self.maxTime and \
               self.minTime < other.maxTime 

    def __contains__(self, other):
        """
        Tests whether the given time point is contained in this interval, 
        either a numeric type or a Point object.
        """
        if hasattr(other, 'minTime') and hasattr(other, 'maxTime'):
            return self.minTime <= other.minTime and \
                   other.maxTime <= self.maxTime
        elif hasattr(other, 'time'):
            return self.minTime <= other.time <= self.maxTime
        else:
            return self.minTime <= other <= self.maxTime

    def bounds(self):
        return (self.minTime, self.maxTime)


class PointTier(object):
    """ 
    Represents Praat PointTiers (also called TextTiers) as list of Points
    (e.g., for point in pointtier). A PointTier is used much like a Python
    set in that it has add/remove methods, not append/extend methods.

    >>> foo = PointTier('foo')
    >>> foo.add(4.0, 'bar')
    >>> foo.add(2.0, 'baz')
    >>> foo
    PointTier(foo, [Point(2.0, baz), Point(4.0, bar)])
    >>> foo.remove(4.0, 'bar')
    >>> foo.add(6.0, 'bar')
    >>> foo
    PointTier(foo, [Point(2.0, baz), Point(6.0, bar)])
    """ 

    def __init__(self, name=None, minTime=0., maxTime=None):
        self.name = name
        self.minTime = minTime
        self.maxTime = maxTime
        self.points = []

    def __str__(self):
        return '<PointTier {0}, {1} points>'.format(self.name, len(self))

    def __repr__(self):
        return 'PointTier({0}, {1})'.format(self.name, self.points)

    def __iter__(self):
        return iter(self.points)

    def __len__(self):
        return len(self.points)

    def __getitem__(self, i):
        return self.points[i]

    def __min__(self):
        return self.minTime

    def __max__(self):
        return self.maxTime

    def add(self, time, mark):
        """ 
        constructs a Point and adds it to the PointTier, maintaining order
        """
        self.addPoint(Point(time, mark))

    def addPoint(self, point):
        if point < self.minTime: 
            raise ValueError(self.minTime) # too early
        if self.maxTime and point > self.maxTime: 
            raise ValueError(self.maxTime) # too late
        i = bisect_left(self.points, point)
        if i < len(self.points) and self.points[i].time == point.time: 
            raise ValueError(point)# we already got one right there
        self.points.insert(i, point)

    def remove(self, time, mark):
        """
        removes a constructed Point i from the PointTier
        """
        self.removePoint(Point(time, mark))

    def removePoint(self, point):
        self.points.remove(point)

    def read(self, f):
        """
        Read the Points contained in the Praat-formated PointTier/TextTier 
        file indicated by string f
        """
        source = readFile(f)
        source.readline() # header junk 
        source.readline()
        source.readline()
        self.minTime = float(source.readline().split()[2])
        self.maxTime = float(source.readline().split()[2])
        for i in xrange(int(source.readline().rstrip().split()[3])):
            source.readline().rstrip() # header
            itim = float(source.readline().rstrip().split()[2])
            imrk = source.readline().rstrip().split()[2].replace('"', '') 
            self.points.append(Point(imrk, itim))

    def write(self, f):
        """
        Write the current state into a Praat-format PointTier/TextTier 
        file. f may be a file object to write to, or a string naming a 
        path for writing
        """
        sink = f if hasattr(f, 'write') else codecs.open(f, 'w', 'UTF-8')
        print >> sink, 'File type = "ooTextFile"'
        print >> sink, 'Object class = "TextTier"'
        print >> sink
        print >> sink, 'xmin = {0}'.format(min(self))
        print >> sink, 'xmax = {0}'.format(max(self))
        print >> sink, 'points: size = {0}'.format(len(self))
        for (i, point) in enumerate(self.points, 1):
            print >> sink, 'points [{0}]:'.format(i)
            print >> sink, '\ttime = {0}'.format(point.time)
            print >> sink, u'\tmark = {0}'.format(point.mark)
        sink.close()

    def bounds(self):
        return (self.minTime, self.maxTime or self.points[-1].time)
    
    # alternative constructor

    @classmethod
    def fromFile(cls, f, name=None):
        pt = cls(name=name)
        pt.read(f)
        return pt


class IntervalTier(object):
    """ 
    Represents Praat IntervalTiers as list of sequence types of Intervals 
    (e.g., for interval in intervaltier). An IntervalTier is used much like a 
    Python set in that it has add/remove methods, not append/extend methods.

    >>> foo = IntervalTier('foo')
    >>> foo.add(0.0, 2.0, 'bar')
    >>> foo.add(2.0, 2.5, 'baz')
    >>> foo
    IntervalTier(foo, [Interval(0.0, 2.0, bar), Interval(2.0, 2.5, baz)])
    >>> foo.remove(0.0, 2.0, 'bar')
    >>> foo
    IntervalTier(foo, [Interval(2.0, 2.5, baz)])
    >>> foo.add(0.0, 1.0, 'bar')
    >>> foo
    IntervalTier(foo, [Interval(0.0, 1.0, bar), Interval(2.0, 2.5, baz)])
    >>> foo.add(1.0, 3.0, 'baz')
    Traceback (most recent call last):
        ...
    ValueError: (Interval(2.0, 2.5, baz), Interval(1.0, 3.0, baz))
    >>> foo.intervalContaining(2.25)
    Interval(2.0, 2.5, baz)
    >>> foo = IntervalTier('foo', maxTime=3.5)
    >>> foo.add(2.7, 3.7, 'bar')
    Traceback (most recent call last):
        ...
    ValueError: 3.5
    >>> foo.add(1.3, 2.4, 'bar')
    >>> foo.add(2.7, 3.3, 'baz')
    >>> temp = foo._fillInTheGaps('') # not for users, but a good quick test
    >>> temp[0]
    Interval(0.0, 1.3, None)
    >>> temp[-1]
    Interval(3.3, 3.5, None)
    >>> temp[2]
    Interval(2.4, 2.7, None)
    """

    def __init__(self, name=None, minTime=0., maxTime=None):
        self.name = name
        self.minTime = minTime
        self.maxTime = maxTime
        self.intervals = []

    def __str__(self):
        return '<IntervalTier {0}, {1} intervals>'.format(self.name, 
                                                          len(self))

    def __repr__(self):
        return 'IntervalTier({0}, {1})'.format(self.name, self.intervals)

    def __iter__(self):
        return iter(self.intervals)

    def __len__(self):
        return len(self.intervals)

    def __getitem__(self, i):
        return self.intervals[i]

    def __min__(self):
        return self.minTime

    def __max__(self):
        return self.maxTime

    def add(self, minTime, maxTime, mark):
        self.addInterval(Interval(minTime, maxTime, mark))

    def addInterval(self, interval):
        if interval.minTime < self.minTime: # too early
            raise ValueError(self.minTime)
        if self.maxTime and interval.maxTime > self.maxTime: # too late
            #raise ValueError, self.maxTime
            raise ValueError(self.maxTime)
        i = bisect_left(self.intervals, interval)
        if i != len(self.intervals) and self.intervals[i] == interval:
            raise ValueError(self.intervals[i])
        self.intervals.insert(i, interval)

    def remove(self, minTime, maxTime, mark):
        self.removeInterval(Interval(minTime, maxTime, mark))

    def removeInterval(self, interval):
        self.intervals.remove(interval)

    def indexContaining(self, time):
        """
        Returns the index of the interval containing the given time point, 
        or None if the time point is outside the bounds of this tier. The 
        argument can be a numeric type, or a Point object.
        """
        i = bisect_left(self.intervals, time) 
        if i != len(self.intervals):
            if self.intervals[i].minTime <= time <= \
                                            self.intervals[i].maxTime:
                return i

    def intervalContaining(self, time):
        """
        Returns the interval containing the given time point, or None if 
        the time point is outside the bounds of this tier. The argument 
        can be a numeric type, or a Point object.
        """
        i = self.indexContaining(time)
        if i: 
            return self.intervals[i]

    def read(self, f):
        """
        Read the Intervals contained in the Praat-formated IntervalTier 
        file indicated by string f
        """
        source = readFile(f)
        source.readline() # header junk 
        source.readline()
        source.readline()
        self.minTime = float(source.readline().split()[2])
        self.maxTime = float(source.readline().split()[2])
        for i in xrange(int(source.readline().rstrip().split()[3])):
            source.readline().rstrip() # header
            imin = float(source.readline().rstrip().split()[2])
            imax = float(source.readline().rstrip().split()[2])
            imrk = source.readline().rstrip().split()[2].replace('"', '')
            self.intervals.append(Interval(imin, imax, imrk))
        source.close()

    def _fillInTheGaps(self, null):
        """
        Returns a pseudo-IntervalTier with the temporal gaps filled in
        """
        prev_t = self.minTime
        output = []
        for interval in self.intervals:
            if prev_t < interval.minTime:
                output.append(Interval(prev_t, interval.minTime, null))
            output.append(interval)
            prev_t = interval.maxTime
        # last interval
        if prev_t < self.maxTime: # also false if maxTime isn't defined
            output.append(Interval(prev_t, self.maxTime, null))
        return output

    def write(self, f, null=''):
        """
        Write the current state into a Praat-format IntervalTier file. f 
        may be a file object to write to, or a string naming a path for 
        writing
        """
        sink = f if hasattr(f, 'write') else open(f, 'w')
        print >> sink, 'File type = "ooTextFile"'
        print >> sink, 'Object class = "IntervalTier"\n'
        print >> sink, 'xmin = {0}'.format(self.minTime)
        print >> sink, 'xmax = {0}'.format(self.maxTime if self.maxTime \
                                          else self.intervals[-1].maxTime)
        # compute the number of intervals and make the empty ones
        output = self._fillInTheGaps(null)
        # write it all out
        print >> sink, 'intervals: size = {0}'.format(len(output))
        for (i, interval) in enumerate(output, 1):
            print >> sink, 'intervals [{0}]'.format(i)
            print >> sink, '\txmin = {0}'.format(interval.minTime)
            print >> sink, '\txmax = {0}'.format(interval.maxTime)
            print >> sink, '\ttext = "{0}"'.format(interval.mark)
        sink.close()

    def bounds(self):
        return self.minTime, self.maxTime or self.intervals[-1].maxTime

    # alternative constructor

    @classmethod
    def fromFile(cls, f, name=None):
        it = cls(name=name)
        it.intervals = []
        it.read(f)
        return it


class TextGrid(object):
    """ 
    Represents Praat TextGrids as list of sequence types of tiers (e.g., 
    for tier in textgrid), and as map from names to tiers (e.g.,
    textgrid['tierName']). Whereas the *Tier classes that make up a 
    TextGrid impose a strict ordering on Points/Intervals, a TextGrid 
    instance is given order by the user. Like a true Python list, there 
    are append/extend methods for a TextGrid.

    >>> foo = TextGrid('foo')
    >>> bar = PointTier('bar')
    >>> bar.add(1.0, 'spam')
    >>> bar.add(2.75, 'eggs')
    >>> baz = IntervalTier('baz')
    >>> baz.add(0.0, 2.5, 'spam')
    >>> baz.add(2.5, 3.5, 'eggs')
    >>> foo.extend([bar, baz])
    >>> foo.append(bar) # now there are two copies of bar in the TextGrid
    >>> foo.minTime
    0.0
    >>> foo.maxTime # nothing
    >>> foo.getFirst('bar')
    PointTier(bar, [Point(1.0, spam), Point(2.75, eggs)])
    >>> foo.getList('bar')[1]
    PointTier(bar, [Point(1.0, spam), Point(2.75, eggs)])
    >>> foo.getNames()
    ['bar', 'baz', 'bar']
    """

    def __init__(self, name=None, minTime=0., maxTime=None):
        """
        Construct a TextGrid instance with the given (optional) name 
        (which is only relevant for MLF stuff). If file is given, it is a 
        string naming the location of a Praat-format TextGrid file from 
        which to populate this instance.
        """
        self.name = name
        self.minTime = minTime
        self.maxTime = maxTime
        self.tiers = []

    def __str__(self):
        return '<TextGrid {0}, {1} Tiers>'.format(self.name, len(self))

    def __repr__(self):
        return 'TextGrid({0}, {1})'.format(self.name, self.tiers)

    def __iter__(self):
        return iter(self.tiers)

    def __len__(self):
        return len(self.tiers)

    def __getitem__(self, i):
        """ 
        Return the ith tier
        """
        return self.tiers[i]

    def getFirst(self, tierName):
        """
        Return the first tier with the given name.
        """
        for t in self.tiers:
            if t.name == tierName:
                return t

    def getList(self, tierName):
        """
        Return a list of all tiers with the given name.
        """
        tiers = []
        for t in self.tiers:
            if t.name == tierName:
                tiers.append(t)
        return tiers

    def getNames(self):
        """
        return a list of the names of the intervals contained in this 
        TextGrid
        """
        return [tier.name for tier in self.tiers]

    def __min__(self):
        return self.minTime

    def __max__(self):
        return self.maxTime

    def append(self, tier):
        if self.maxTime and tier.maxTime > self.maxTime: 
            raise ValueError(self.maxTime) # too late
        self.tiers.append(tier)

    def extend(self, tiers):
        if min([t.minTime for t in tiers]) < self.minTime:
            raise ValueError(self.minTime) # too early
        if self.maxTime and max([t.minTime for t in tiers]) > self.maxTime:
            raise ValueError(self.maxTime) # too late
        self.tiers.extend(tiers)

    def pop(self, i=None):
        """
        Remove and return tier at index i (default last). Will raise 
        IndexError if TextGrid is empty or index is out of range.
        """
        return (self.tiers.pop(i) if i else self.tiers.pop())

    @staticmethod
    def _getMark(text):
        """
        Get the "mark" text on a line. Since Praat doesn't prevent you 
        from using your platform's newline character in "text" fields, we
        read until we find a match. Regression tests are in `RWtests.py`.
        """
        m = None
        my_line = ''
        while True:
            my_line += text.readline()
            m = re.search(r'(\S+)\s(=)\s(".*")', my_line, 
                          re.DOTALL)
            if m != None:
                break
        return m.groups()[2][1:-1]

    def read(self, f):
        """
        Read the tiers contained in the Praat-formated TextGrid file 
        indicated by string f
        """
        source = readFile(f)
        source.readline() # header junk
        source.readline() # header junk
        source.readline() # header junk
        self.minTime = round(float(source.readline().split()[2]), 5)
        self.maxTime = round(float(source.readline().split()[2]), 5)
        source.readline() # more header junk
        m = int(source.readline().rstrip().split()[2]) # will be self.n
        source.readline()
        for i in xrange(m): # loop over grids
            source.readline()
            if source.readline().rstrip().split()[2] == '"IntervalTier"': 
                inam = source.readline().rstrip().split(' = ')[1].strip('"')
                imin = round(float(source.readline().rstrip().split()[2]), 5)
                imax = round(float(source.readline().rstrip().split()[2]), 5)
                itie = IntervalTier(inam)
                for j in xrange(int(source.readline().rstrip().split()[3])):
                    source.readline().rstrip().split() # header junk
                    jmin = round(float(source.readline().rstrip().split()[2]), 5)
                    jmax = round(float(source.readline().rstrip().split()[2]), 5)
                    jmrk = self._getMark(source)
                    if jmin < jmax: # non-null
                        itie.addInterval(Interval(jmin, jmax, jmrk))
                self.append(itie)
            else: # pointTier
                inam = source.readline().rstrip().split(' = ')[1].strip('"')
                imin = round(float(source.readline().rstrip().split()[2]), 5)
                imax = round(float(source.readline().rstrip().split()[2]), 5)
                itie = PointTier(inam)
                n = int(source.readline().rstrip().split()[3])
                for j in xrange(n):
                    source.readline().rstrip() # header junk
                    jtim = round(float(source.readline().rstrip().split()[2]),
                                                                           5)
                    jmrk = source.readline().rstrip().split()[2][1:-1]
                    itie.addPoint(Point(jtim, jmrk))
                self.append(itie)
        source.close()

    def write(self, f, null=''):
        """
        Write the current state into a Praat-format TextGrid file. f may 
        be a file object to write to, or a string naming a path to open 
        for writing.
        """
        sink = f if hasattr(f, 'write') else codecs.open(f, 'w', 'UTF-8')
        print >> sink, 'File type = "ooTextFile"'
        print >> sink, 'Object class = "TextGrid"\n'
        print >> sink, 'xmin = {0}'.format(self.minTime)
        # compute max time
        maxT = self.maxTime
        if not maxT:
            maxT = max([t.maxTime if t.maxTime else t[-1].maxTime \
                                               for t in self.tiers])
        print >> sink, 'xmax = {0}'.format(maxT)
        print >> sink, 'tiers? <exists>'
        print >> sink, 'size = {0}'.format(len(self))
        print >> sink, 'item []:'
        for (i, tier) in enumerate(self.tiers, 1):
            print >> sink, '\titem [{0}]:'.format(i)
            if tier.__class__ == IntervalTier: 
                print >> sink, '\t\tclass = "IntervalTier"'
                print >> sink, '\t\tname = "{0}"'.format(tier.name)
                print >> sink, '\t\txmin = {0}'.format(tier.minTime)
                print >> sink, '\t\txmax = {0}'.format(maxT)
                # compute the number of intervals and make the empty ones
                output = tier._fillInTheGaps(null)
                print >> sink, '\t\tintervals: size = {0}'.format(
                                                           len(output))
                for (j, interval) in enumerate(output, 1):
                    print >> sink, '\t\t\tintervals [{0}]:'.format(j)
                    print >> sink, '\t\t\t\txmin = {0}'.format(
                                                        interval.minTime)
                    print >> sink, '\t\t\t\txmax = {0}'.format(
                                                        interval.maxTime)
                    print >> sink, u'\t\t\t\ttext = "{0}"'.format(
                                                        interval.mark)
            elif tier.__class__ == PointTier: # PointTier
                print >> sink, '\t\tclass = "TextTier"'
                print >> sink, '\t\tname = "{0}"'.format(tier.name)
                print >> sink, '\t\txmin = {0}'.format(min(tier))
                print >> sink, '\t\txmax = {0}'.format(max(tier))
                print >> sink, '\t\tpoints: size = {0}'.format(len(tier))
                for (k, point) in enumerate(tier, 1):
                    print >> sink, '\t\t\tpoints [{0}]:'.format(k)
                    print >> sink, '\t\t\t\ttime = {0}'.format(point.time)
                    print >> sink, u'\t\t\t\tmark = "{0}"'.format(
                                                           point.mark)
        sink.close()

    # alternative constructor

    @classmethod
    def fromFile(cls, f, name=None):
        tg = cls(name=name)
        tg.read(f)
        return tg


class MLF(object):
    """
    Read in a HTK .mlf file generated with HVite -o SM and turn it into a 
    list of TextGrids. The resulting class can be iterated over to give 
    one TextGrid at a time, or the write(prefix='') class method can be 
    used to write all the resulting TextGrids into separate files.

    Unlike other classes, this is always initialized from a text file.
    """

    def __init__(self, f, samplerate=10e6):
        self.grids = []
        self.read(f, samplerate)

    def __iter__(self):
        return iter(self.grids)

    def __str__(self):
        return '<MLF, {0} TextGrids>'.format(len(self))

    def __repr__(self):
        return 'MLF({0})'.format(self.grids)

    def __len__(self):
        return len(self.grids)

    def __getitem__(self, i):
        """ 
        Return the ith TextGrid
        """
        return self.grids[i]

    def read(self, f, samplerate):
        source = open(f, 'r') # HTK returns ostensible ASCII
        samplerate = float(samplerate)
        source.readline() # header
        while True: # loop over text
            name = re.match('\"(.*)\"', source.readline().rstrip())
            if name:
                name = name.groups()[0]
                grid = TextGrid(name)
                phon = IntervalTier(name='phones')
                word = IntervalTier(name='words')
                wmrk = ''
                wsrt = 0.
                wend = 0.
                while 1: # loop over the lines in each grid
                    line = source.readline().rstrip().split()
                    if len(line) == 4: # word on this baby
                        pmin = round(float(line[0]) / samplerate, 5)
                        pmax = round(float(line[1]) / samplerate, 5)
                        if pmin == pmax:
                            raise ValueError('null duration interval')
                        phon.add(pmin, pmax, line[2])
                        if wmrk:
                            word.add(wsrt, wend, wmrk)
                        wmrk = decode(line[3])
                        wsrt = pmin
                        wend = pmax
                    elif len(line) == 3: # just phone
                        pmin = round(float(line[0]) / samplerate, 5)
                        pmax = round(float(line[1]) / samplerate, 5)
                        if line[2] == 'sp' and pmin != pmax:
                            if wmrk:
                                word.add(wsrt, wend, wmrk)
                            wmrk = decode(line[2])
                            wsrt = pmin
                            wend = pmax
                        elif pmin != pmax:
                            phon.add(pmin, pmax, line[2])
                        wend = pmax
                    else: # it's a period
                        word.add(wsrt, wend, wmrk)
                        self.grids.append(grid)
                        break
                grid.append(phon)
                grid.append(word)
            else:
                source.close()
                break

    def write(self, prefix=''):
        """ 
        Write the current state into Praat-formatted TextGrids. The 
        filenames that the output is stored in are taken from the HTK 
        label files. If a string argument is given, then the any prefix in 
        the name of the label file (e.g., "mfc/myLabFile.lab"), it is 
        truncated and files are written to the directory given by the 
        prefix. An IOError will result if the folder does not exist.
    
        The number of TextGrids is returned.
        """
        for grid in self.grids:
            (junk, tail) = os.path.split(grid.name)
            (root, junk) = os.path.splitext(tail)
            my_path = os.path.join(prefix, root + '.TextGrid')
            grid.write(codecs.open(my_path, 'w', 'UTF-8'))
        return len(self.grids)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
