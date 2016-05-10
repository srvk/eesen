silence_threshold = -15
minimum_pause_duration = 0.3
minimum_pitch = 75
maximum_pitch = 500
directory$ = "./"
resultfile$ = "./praat.res"

# shorten default variable names
silencedb = 'silence_threshold'
minpause = 'minimum_pause_duration'

# read files
Create Strings as file list... list 'directory$'/*.wav
numberOfFiles = Get number of strings
for ifile to numberOfFiles
   select Strings list
   fileName$ = Get string... ifile
   nowarn Read from file... 'directory$'/'fileName$'

# use object ID
   soundname$ = selected$("Sound")
   soundid = selected("Sound")

   originaldur = Get total duration
   # allow non-zero starting time
   bt = Get starting time

   # Use intensity to get threshold
   To Intensity... 50 0 yes
   intid = selected("Intensity")

   # estimate noise floor
   minint = Get minimum... 0 0 Parabolic
   # estimate noise max
   maxint = Get maximum... 0 0 Parabolic
   #get .99 quantile to get maximum (without influence of non-speech sound bursts)
   max99int = Get quantile... 0 0 0.99

   threshold2 = maxint - max99int
   threshold3 = silencedb - threshold2
   state = 1

  # get pauses (silences) and speakingtime, the minimum of sounding should be longer to prevent f0 from being undefined
   nowarn To TextGrid (silences)... threshold3 minpause 0.05 silent sounding
   # the average rate is about 160 words a minute
   textgridid = selected("TextGrid")
   # silencetierid = Extract tier... 1
   # silencetableid = Down to TableOfReal... sounding
   Down to Table... no 2 no no
   tableid = selected("Table")
   rownum = Search column... text sounding
   rownum$ = fixed$ (rownum, 2) 
   echo 'rownum$'
   if rownum <> 0
   beginsound = Get value... rownum tmin
   endsound = Get value... rownum tmax
   speakingdur = 'endsound' - 'beginsound'
   elif rownum = 0
   beginsound = 0
   endsound = originaldur
   speakingdur = 'endsound' - 'beginsound'
   endif

#if sth wrong, goes to state 2
   if beginsound = 0 and endsound < 0.15
   state = 2
    select 'tableid'
    plus 'textgridid'
    Remove

   select intid

   nowarn To TextGrid (silences)... threshold3 minpause 0.11 silent sounding
   # the average rate is about 160 words a minute
   textgridid = selected("TextGrid")
   # silencetierid = Extract tier... 1
   # silencetableid = Down to TableOfReal... sounding
   Down to Table... no 2 no no
   tableid = selected("Table")
   rownum = Search column... text sounding
   rownum$ = fixed$ (rownum, 2) 
   echo 'rownum$'
   if rownum <> 0
   beginsound = Get value... rownum tmin
   endsound = Get value... rownum tmax
   speakingdur = 'endsound' - 'beginsound'
   elif rownum = 0
   beginsound = 0
   endsound = originaldur
   speakingdur = 'endsound' - 'beginsound'
   endif

  endif

    print 'endsound'
    print 'originaldur'

#if sth wrong, goes to state 3
   if beginsound = 0 and endsound > (0.9 * originaldur)
    state = 3 
   
    select 'tableid'
    plus 'textgridid'
    Remove

   silencedb = -7
   select intid

   threshold3 = silencedb - threshold2

   nowarn To TextGrid (silences)... threshold3 minpause 0.11 silent sounding
   # the average rate is about 160 words a minute
   textgridid = selected("TextGrid")
   # silencetierid = Extract tier... 1
   # silencetableid = Down to TableOfReal... sounding
   Down to Table... no 2 no no
   tableid = selected("Table")
   rownum = Search column... text sounding
   rownum$ = fixed$ (rownum, 2) 
   echo 'rownum$'
   if rownum <> 0
   beginsound = Get value... rownum tmin
   endsound = Get value... rownum tmax
   speakingdur = 'endsound' - 'beginsound'
   elif rownum = 0
   beginsound = 0
   endsound = originaldur
   speakingdur = 'endsound' - 'beginsound'
   endif

  endif

  endsound = 'originaldur' - 'beginsound'
  speakingdur = 'endsound' - 'beginsound'

# end of all the if strcutures

# ontain pitch, formants, and intensity
   threshold3$ = fixed$ (threshold3, 2)
   print 'threshold3$'
   select soundid
   To Pitch... 0.01 'minimum_pitch' 'maximum_pitch'
   pitchid = selected("Pitch")

   select soundid
   To Formant (burg)... 0.0 5 8000 0.025 50
   formantid = selected("Formant")

  select pitchid
  f0 = Get mean... 'beginsound' 'endsound' Hertz
  f0$ = fixed$ (f0, 2)    

  select intid
  intensity = Get mean... 'beginsound' 'endsound' energy
  #convert to string to output
  intensity$ = fixed$ (intensity, 2)

  select formantid
  f1 = Get mean... 1 'beginsound' 'endsound' Hertz
  f2 = Get mean... 2 'beginsound' 'endsound' Hertz
  f3 = Get mean... 3 'beginsound' 'endsound' Hertz
  f1$ = fixed$ (f1, 2) 
  f2$ = fixed$ (f2, 2) 
  f3$ = fixed$ (f3, 2) 
  state$ = fixed$ (state, 2) 
 
 # clean up the screen before next sound file is opened
  select 'intid'
  plus 'formantid'
  plus 'pitchid'
  plus 'tableid'
  # plus 'silencetierid'
  # plus 'silencetableid'
  plus 'soundid'
  plus 'textgridid'
  Remove

# compose a line and write it to a file
# The '0' is a fake gradelevel used by error analysis webapp

  resultline$ = "'fileName$'  'originaldur:2' 'speakingdur:2' 'beginsound:2'  'endsound:2' 0 'intensity$' 'f0$'   'f1$'  'f2$'  'f3$' 'newline$'"
  fileappend "'resultfile$'" 'resultline$' 
 
endfor
