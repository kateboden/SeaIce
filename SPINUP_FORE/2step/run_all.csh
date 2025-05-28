#!/bin/csh

set mem = 1
while ($mem <= 80)
    echo "running $mem"
    set inst_string = `printf %04d $mem`
    cd mem${inst_string}
    # Copy analysis to restart
    cp restart/analysis.nc restart/restart.nc
    # Run icepack
    ./icepack >&! out.txt &
    cd ../
    @ mem = $mem + 1
end
wait

set mem = 1
while ($mem <= 80)
    set inst_string = `printf %04d $mem`
    cd mem${inst_string}
    if ( { grep -q ABORTED out.txt } ) then
        mv restart/restart.nc restart/bad_restart.nc
    else
        cd restart/
        # change the output file name to forecast.nc
        mv iced.* forecast.nc
        cp forecast.nc analysis.nc
        cd ../
    endif
    rm history/*
    cd ../
    @ mem = $mem + 1
end
