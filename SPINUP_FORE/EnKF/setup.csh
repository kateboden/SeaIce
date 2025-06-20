#!/bin/csh

set sim_length = 24

set rsno_params  = (-1.03125 0.46875 1.09375 		  0.34375	0.34375	1.96875	-1.90625	1.28125	1.46875	1.96875 \
0.59375	-0.90625	-1.71875	0.71875	-1.65625	-0.03125	0.09375	-0.71875	-0.53125	-0.03125 \
-0.40625	-1.90625	1.28125	1.71875	1.34375	-1.40625	-0.90625	-1.71875	-1.53125	0.96875 \
1.59375	0.09375	-0.71875	-0.28125	-0.65625	0.96875	1.09375	0.28125	0.46875	-1.03125 \
-1.65625	-1.28125	1.03125	0.28125	-1.90625	1.21875	-1.65625	-1.46875	1.71875	1.71875 \
0.34375	-1.65625	-0.96875	-1.53125	0.09375	-0.78125	0.34375	0.53125	-0.28125	-0.28125 \
-0.65625	-0.65625	0.03125	-0.53125	-0.90625	-1.78125	-0.65625	1.53125	-1.28125	0.71875 \
1.34375	1.34375	-1.96875	1.46875	1.09375	0.21875	1.34375	-0.46875	0.71875	-1.28125 )

#########################
set rsnw_mlt = ( 916 1947 2377   	1861	1861	2979	314	2506	2635 2979\
2033	1002	443	2119	486	1604	1689	1131	1260 1604 \
1346	314	2506	2807	2549	658	1002	443	572  2291\
2721	1689	1131	1432	1174	2291	2377	1818	1947 916\
486	744	2334	1818	314	2463	486	615	2807 2807\
1861	486	959	572	1689	1088	1861	1990	1432 1432\
1174	1174	1646	1260	1002	400	1174	2678	744  2119\
2549	2549	271	2635	2377	1775	2549	1303	2119 744)

#########################
set ksno_params = ( 0.3848 0.5160 0.5707  	0.5051	0.5051	0.6473	0.3082	0.5871	0.6035	0.6473 \
0.5270	0.3957	0.3246	0.5379	0.3301	0.4723	0.4832	0.4121	0.4285	0.4723 \
0.4395	0.3082	0.5871	0.6254	0.5926	0.3520	0.3957	0.3246	0.3410	0.5598\
0.6145	0.4832	0.4121	0.4504	0.4176	0.5598	0.5707	0.4996	0.5160	0.3848 \
0.3301	0.3629	0.5652	0.4996	0.3082	0.5816	0.3301	0.3465	0.6254	0.6254\
0.5051	0.3301	0.3902	0.3410	0.4832	0.4066	0.5051	0.5215	0.4504	0.4504 \
0.4176	0.4176	0.4777	0.4285	0.3957	0.3191	0.4176	0.6090	0.3629	0.5379 \
0.5926	0.5926	0.3027	0.6035	0.5707	0.4941	0.5926	0.4340	0.5379	0.3629 )

set mem = 1
while ($mem <= 80)
    echo "setting up member $mem"
    set inst_string = `printf %04d $mem`
    mkdir -p mem${inst_string}
    mkdir -p mem${inst_string}/history
    mkdir -p mem${inst_string}/restart
    cd mem${inst_string}
    set restart_flag = .true.
    cat >! script.sed << EOF
    /SIM_LEN/c\
    npt              = $sim_length
    /RESTART_LOGIC/c\
    restart           = $restart_flag
    /KSNO/c\
    ksno              = ${ksno_params[$mem]}
    /ALBV/c\
    albicev         = 0.7 
    /ALBI/c\
    albicei         = 0.35
    /RSNW/c\
    R_snw           = ${rsno_params[$mem]}
    /RMLT/c\
    rsnw_mlt        = ${rsnw_mlt[$mem]}
    /MEMA/c\
    atm_data_file   = 'ATM_FORCING_${inst_string}.txt'
    /MEMO/c\
    ocn_data_file   = 'OCN_FORCING_${inst_string}.txt'  
EOF
    sed -f script.sed ../icepack_in.template >! icepack_in
    rm script.sed
    cp ../icepack.settings .
    cp /Users/kabo1917/icepack-dirs/runs/case2/icepack .
    cd ../    
    @ mem = $mem + 1
end

