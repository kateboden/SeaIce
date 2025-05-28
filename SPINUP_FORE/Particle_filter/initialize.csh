#!/bin/csh

rm mem*/restart/iced.*

set mem = 1
while ($mem <= 80)
    echo "initializing $mem"
    set inst_string = `printf %04d $mem`
    cd mem${inst_string}
    cp ../../initial_conditions/iced.2011-01-02-00000.nc_${inst_string} restart/forecast.nc 
    cp ../../initial_conditions/iced.2011-01-02-00000.nc_${inst_string} restart/analysis.nc 
    cd ../
    @ mem = $mem + 1
end
