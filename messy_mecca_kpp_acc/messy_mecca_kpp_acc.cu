/*************************************************************
 * 
 *    kpp_integrate_cuda_prototype.cu
 *    Prototype file for kpp CUDA kernel
 *
 *    Copyright 2016 The Cyprus Institute 
 *
 *    Developers: Michail Alvanos - m.alvanos@cyi.ac.cy
 *                Giannis Ashiotis
 *                Theodoros Christoudias - christoudias@cyi.ac.cy
 *
 ********************************************************************/

#include <stdio.h>
#include <unistd.h>
#include "cuda.h"

#define NSPEC 142
#define NVAR 139
#define NFIX 3
#define NREACT 310
#define LU_NONZERO 1486
#define NBSIZE 523
#define BLOCKSIZE 64

//#define MAX_VL_GLO 12288 /* elements that will pass in each call */

#define REDUCTION_SIZE_1 64
#define REDUCTION_SIZE_2 32

#define R_gas 8.3144621
#define N_A 6.02214129e+23
#define atm2Pa 101325.0

#define ip_O2 0
#define ip_O3P 1
#define ip_O1D 2
#define ip_H2O2 3
#define ip_NO2 4
#define ip_NO2O 5
#define ip_NOO2 6
#define ip_N2O5 7
#define ip_HNO3 8
#define ip_HNO4 9
#define ip_PAN 10
#define ip_HONO 11
#define ip_CH3OOH 12
#define ip_COH2 13
#define ip_CHOH 14
#define ip_CH3CO3H 15
#define ip_CH3CHO 16
#define ip_CH3COCH3 17
#define ip_MGLYOX 18
#define ip_HOCl 19
#define ip_OClO 20
#define ip_Cl2O2 21
#define ip_ClNO3 22
#define ip_ClNO2 23
#define ip_Cl2 24
#define ip_BrO 25
#define ip_HOBr 26
#define ip_BrCl 27
#define ip_BrNO3 28
#define ip_BrNO2 29
#define ip_Br2 30
#define ip_CCl4 31
#define ip_CH3Cl 32
#define ip_CH3CCl3 33
#define ip_CFCl3 34
#define ip_CF2Cl2 35
#define ip_CH3Br 36
#define ip_CF2ClBr 37
#define ip_CF3Br 38
#define ip_CH3I 39
#define ip_C3H7I 40
#define ip_CH2ClI 41
#define ip_CH2I2 42
#define ip_IO 43
#define ip_HOI 44
#define ip_I2 45
#define ip_ICl 46
#define ip_IBr 47
#define ip_INO2 48
#define ip_INO3 49
#define ip_SO2 50
#define ip_SO3 51
#define ip_OCS 52
#define ip_CS2 53
#define ip_H2O 54
#define ip_N2O 55
#define ip_NO 56
#define ip_CO2 57
#define ip_HCl 58
#define ip_CHCl2Br 59
#define ip_CHClBr2 60
#define ip_CH2ClBr 61
#define ip_CH2Br2 62
#define ip_CHBr3 63
#define ip_SF6 64
#define ip_NO3NOO 65
#define ip_ClONO2 66
#define ip_MACR 67
#define ip_MVK 68
#define ip_GLYOX 69
#define ip_HOCH2CHO 70
#define ip_CH4 71
#define ip_O2_b1b2 72
#define ip_O2_b1 73
#define ip_O2_b2 74
#define ip_O3PO1D 75
#define ip_O3Pp 76
#define ip_H2O1D 77
#define ip_N2 78
#define ip_N2_b1 79
#define ip_N2_b2 80
#define ip_N2_b3 81
#define ip_NN2D 82
#define ip_NOp 83
#define ip_Op_em 84
#define ip_O2p_em 85
#define ip_Op_O_em 86
#define ip_N2p_em 87
#define ip_Np_N_em 88
#define ip_Np_N2D_em 89
#define ip_N_N2D_em 90
#define ip_Op_em_b 91
#define ip_se_O2_b1 92
#define ip_se_O2_b2 93
#define ip_se_N2_b1 94
#define ip_se_N2_b2 95
#define ip_se_N2_b3 96
#define ip_se_N2_b4 97
#define ip_se_Op_em 98
#define ip_O2_aurq 99
#define ip_N2_aurq 100
#define ip_H2SO4 101
#define ip_C3O2 102
#define ip_CH3NO3 103
#define ip_CH3O2NO2 104
#define ip_CH3ONO 105
#define ip_CH3O2 106
#define ip_HCOOH 107
#define ip_HO2NO2 108
#define ip_OHNO3 109
#define ip_qqqdummy 110
#define ip_CH3OCl 111
#define ip_MEO2NO2 112
#define ip_CHF2Cl 113
#define ip_F113 114
#define ip_C2H5NO3 115
#define ip_NOA 116
#define ip_MEKNO3 117
#define ip_BENZAL 118
#define ip_HOPh3Me2NO2 119
#define ip_HOC6H4NO2 120
#define ip_CH3CHO2VINY 121
#define ip_CH3COCO2H 122
#define ip_IPRCHO2HCO 123
#define ip_C2H5CHO2HCO 124
#define ip_C2H5CHO2ENOL 125
#define ip_C3H7CHO2HCO 126
#define ip_C3H7CHO2VINY 127
#define ip_PeDIONE24 128
#define ip_PINAL2HCO 129
#define ip_PINAL2ENOL 130
#define ip_CF2ClCFCl2 131
#define ip_CH3CFCl2 132
#define ip_CF3CF2Cl 133
#define ip_CF2ClCF2Cl 134
#define ip_CHCl3 135
#define ip_CH2Cl2 136
#define ip_HO2 137
#define ip_ClO 138

#define ind_BrNO2 0
#define ind_CF2ClBr 1
#define ind_CF3Br 2
#define ind_CH3I 3
#define ind_O3s 4
#define ind_CF2ClBr_c 5
#define ind_CF3Br_c 6
#define ind_LCARBON 7
#define ind_LFLUORINE 8
#define ind_LCHLORINE 9
#define ind_CH3SO3H 10
#define ind_H2SO4 11
#define ind_NO3m_cs 12
#define ind_Hp_cs 13
#define ind_Dummy 14
#define ind_CFCl3_c 15
#define ind_CF2Cl2_c 16
#define ind_N2O_c 17
#define ind_CH3CCl3_c 18
#define ind_LO3s 19
#define ind_LossHO2 20
#define ind_LossO1D 21
#define ind_LossO3 22
#define ind_LossO3Br 23
#define ind_LossO3Cl 24
#define ind_LossO3H 25
#define ind_LossO3N 26
#define ind_LossO3O 27
#define ind_LossO3R 28
#define ind_LossOH 29
#define ind_ProdHO2 30
#define ind_ProdLBr 31
#define ind_ProdLCl 32
#define ind_ProdMeO2 33
#define ind_ProdO3 34
#define ind_ProdRO2 35
#define ind_ProdSBr 36
#define ind_ProdSCl 37
#define ind_BIACET 38
#define ind_Cl2O2 39
#define ind_NC4H10 40
#define ind_CCl4 41
#define ind_CF2Cl2 42
#define ind_CFCl3 43
#define ind_CH2Br2 44
#define ind_CHBr3 45
#define ind_CH3SO3 46
#define ind_NH3 47
#define ind_C2H6 48
#define ind_C3H8 49
#define ind_ClNO2 50
#define ind_OClO 51
#define ind_CH2ClBr 52
#define ind_CH3Br 53
#define ind_CHCl2Br 54
#define ind_CHClBr2 55
#define ind_SO2 56
#define ind_CH3CCl3 57
#define ind_NACA 58
#define ind_N 59
#define ind_N2O 60
#define ind_NH2OH 61
#define ind_IC3H7NO3 62
#define ind_CH3CO3H 63
#define ind_MPAN 64
#define ind_DMSO 65
#define ind_ISOOH 66
#define ind_LHOC3H6OOH 67
#define ind_LMEKOOH 68
#define ind_IC3H7OOH 69
#define ind_NHOH 70
#define ind_C2H5OOH 71
#define ind_HYPERACET 72
#define ind_HNO4 73
#define ind_CH3CO2H 74
#define ind_CH3Cl 75
#define ind_HONO 76
#define ind_PAN 77
#define ind_HCOOH 78
#define ind_LC4H9OOH 79
#define ind_Cl2 80
#define ind_CH3SO2 81
#define ind_MVKOOH 82
#define ind_N2O5 83
#define ind_NH2O 84
#define ind_MEK 85
#define ind_CH3COCH3 86
#define ind_HNO 87
#define ind_H2O2 88
#define ind_CH3OH 89
#define ind_BrCl 90
#define ind_ISON 91
#define ind_NH2 92
#define ind_IC3H7O2 93
#define ind_CH3COCH2O2 94
#define ind_CO 95
#define ind_MGLYOX 96
#define ind_H2 97
#define ind_CH4 98
#define ind_LMEKO2 99
#define ind_Br2 100
#define ind_HNO3 101
#define ind_LC4H9O2 102
#define ind_C2H4 103
#define ind_CH3OOH 104
#define ind_BrNO3 105
#define ind_C5H8 106
#define ind_C3H6 107
#define ind_ACETOL 108
#define ind_ISO2 109
#define ind_MVK 110
#define ind_LC4H9NO3 111
#define ind_HOCl 112
#define ind_MVKO2 113
#define ind_DMS 114
#define ind_LHOC3H6O2 115
#define ind_ClNO3 116
#define ind_C2H5O2 117
#define ind_HOBr 118
#define ind_CH3CHO 119
#define ind_O1D 120
#define ind_CH3CO3 121
#define ind_H 122
#define ind_HBr 123
#define ind_O3 124
#define ind_CH3O2 125
#define ind_OH 126
#define ind_Cl 127
#define ind_H2O 128
#define ind_Br 129
#define ind_HCHO 130
#define ind_O3P 131
#define ind_BrO 132
#define ind_NO 133
#define ind_ClO 134
#define ind_NO2 135
#define ind_NO3 136
#define ind_HO2 137
#define ind_HCl 138
#define ind_O2 139
#define ind_N2 140
#define ind_CO2 141
#define ind_H2OH2O -1
#define ind_N2D -1
#define ind_LNITROGEN -1
#define ind_CH2OO -1
#define ind_CH2OOA -1
#define ind_CH3 -1
#define ind_CH3O -1
#define ind_HOCH2O2 -1
#define ind_HOCH2OH -1
#define ind_HOCH2OOH -1
#define ind_CH3NO3 -1
#define ind_CH3O2NO2 -1
#define ind_CH3ONO -1
#define ind_CN -1
#define ind_HCN -1
#define ind_HOCH2O2NO2 -1
#define ind_NCO -1
#define ind_C2H2 -1
#define ind_C2H5OH -1
#define ind_CH2CHOH -1
#define ind_CH2CO -1
#define ind_CH3CHOHO2 -1
#define ind_CH3CHOHOOH -1
#define ind_CH3CO -1
#define ind_ETHGLY -1
#define ind_GLYOX -1
#define ind_HCOCH2O2 -1
#define ind_HCOCO -1
#define ind_HCOCO2H -1
#define ind_HCOCO3 -1
#define ind_HCOCO3H -1
#define ind_HOCH2CH2O -1
#define ind_HOCH2CH2O2 -1
#define ind_HOCH2CHO -1
#define ind_HOCH2CO -1
#define ind_HOCH2CO2H -1
#define ind_HOCH2CO3 -1
#define ind_HOCH2CO3H -1
#define ind_HOCHCHO -1
#define ind_HOOCH2CHO -1
#define ind_HOOCH2CO2H -1
#define ind_HOOCH2CO3 -1
#define ind_HOOCH2CO3H -1
#define ind_HYETHO2H -1
#define ind_C2H5NO3 -1
#define ind_C2H5O2NO2 -1
#define ind_CH3CN -1
#define ind_ETHOHNO3 -1
#define ind_NCCH2O2 -1
#define ind_NO3CH2CHO -1
#define ind_NO3CH2CO3 -1
#define ind_NO3CH2PAN -1
#define ind_PHAN -1
#define ind_ALCOCH2OOH -1
#define ind_C2H5CHO -1
#define ind_C2H5CO2H -1
#define ind_C2H5CO3 -1
#define ind_C2H5CO3H -1
#define ind_C33CO -1
#define ind_CH3CHCO -1
#define ind_CH3COCO2H -1
#define ind_CH3COCO3 -1
#define ind_CH3COCO3H -1
#define ind_CHOCOCH2O2 -1
#define ind_HCOCH2CHO -1
#define ind_HCOCH2CO2H -1
#define ind_HCOCH2CO3 -1
#define ind_HCOCH2CO3H -1
#define ind_HCOCOCH2OOH -1
#define ind_HOC2H4CO2H -1
#define ind_HOC2H4CO3 -1
#define ind_HOC2H4CO3H -1
#define ind_HOCH2COCH2O2 -1
#define ind_HOCH2COCH2OOH -1
#define ind_HOCH2COCHO -1
#define ind_HYPROPO2 -1
#define ind_HYPROPO2H -1
#define ind_IPROPOL -1
#define ind_NC3H7O2 -1
#define ind_NC3H7OOH -1
#define ind_NPROPOL -1
#define ind_PROPENOL -1
#define ind_C32OH13CO -1
#define ind_C3DIALO2 -1
#define ind_C3DIALOOH -1
#define ind_HCOCOHCO3 -1
#define ind_HCOCOHCO3H -1
#define ind_METACETHO -1
#define ind_C3PAN1 -1
#define ind_C3PAN2 -1
#define ind_CH3COCH2O2NO2 -1
#define ind_NC3H7NO3 -1
#define ind_NOA -1
#define ind_PPN -1
#define ind_PR2O2HNO3 -1
#define ind_PRONO3BO2 -1
#define ind_PROPOLNO3 -1
#define ind_HCOCOHPAN -1
#define ind_BIACETO2 -1
#define ind_BIACETOH -1
#define ind_BIACETOOH -1
#define ind_BUT1ENE -1
#define ind_BUT2OLO -1
#define ind_BUT2OLO2 -1
#define ind_BUT2OLOOH -1
#define ind_BUTENOL -1
#define ind_C312COCO3 -1
#define ind_C312COCO3H -1
#define ind_C3H7CHO -1
#define ind_C413COOOH -1
#define ind_C44O2 -1
#define ind_C44OOH -1
#define ind_C4CODIAL -1
#define ind_CBUT2ENE -1
#define ind_CH3COCHCO -1
#define ind_CH3COCHO2CHO -1
#define ind_CH3COCOCO2H -1
#define ind_CH3COOHCHCHO -1
#define ind_CHOC3COO2 -1
#define ind_CO23C3CHO -1
#define ind_CO2C3CHO -1
#define ind_CO2H3CHO -1
#define ind_CO2H3CO2H -1
#define ind_CO2H3CO3 -1
#define ind_CO2H3CO3H -1
#define ind_EZCH3CO2CHCHO -1
#define ind_EZCHOCCH3CHO2 -1
#define ind_HCOCCH3CHOOH -1
#define ind_HCOCCH3CO -1
#define ind_HCOCO2CH3CHO -1
#define ind_HMAC -1
#define ind_HO12CO3C4 -1
#define ind_HVMK -1
#define ind_IBUTALOH -1
#define ind_IBUTDIAL -1
#define ind_IBUTOLBO2 -1
#define ind_IBUTOLBOOH -1
#define ind_IC4H10 -1
#define ind_IC4H9O2 -1
#define ind_IC4H9OOH -1
#define ind_IPRCHO -1
#define ind_IPRCO3 -1
#define ind_IPRHOCO2H -1
#define ind_IPRHOCO3 -1
#define ind_IPRHOCO3H -1
#define ind_MACO2 -1
#define ind_MACO2H -1
#define ind_MACO3 -1
#define ind_MACO3H -1
#define ind_MACR -1
#define ind_MACRO -1
#define ind_MACRO2 -1
#define ind_MACROH -1
#define ind_MACROOH -1
#define ind_MBOOO -1
#define ind_MEPROPENE -1
#define ind_MPROPENOL -1
#define ind_PERIBUACID -1
#define ind_TBUT2ENE -1
#define ind_TC4H9O2 -1
#define ind_TC4H9OOH -1
#define ind_BZFUCO -1
#define ind_BZFUO2 -1
#define ind_BZFUONE -1
#define ind_BZFUOOH -1
#define ind_CO14O3CHO -1
#define ind_CO14O3CO2H -1
#define ind_CO2C4DIAL -1
#define ind_EPXC4DIAL -1
#define ind_EPXDLCO2H -1
#define ind_EPXDLCO3 -1
#define ind_EPXDLCO3H -1
#define ind_HOCOC4DIAL -1
#define ind_MALANHY -1
#define ind_MALANHYO2 -1
#define ind_MALANHYOOH -1
#define ind_MALDALCO2H -1
#define ind_MALDALCO3H -1
#define ind_MALDIAL -1
#define ind_MALDIALCO3 -1
#define ind_MALDIALO2 -1
#define ind_MALDIALOOH -1
#define ind_MALNHYOHCO -1
#define ind_MECOACEOOH -1
#define ind_MECOACETO2 -1
#define ind_BUT2OLNO3 -1
#define ind_C312COPAN -1
#define ind_C4PAN5 -1
#define ind_IBUTOLBNO3 -1
#define ind_IC4H9NO3 -1
#define ind_MACRN -1
#define ind_MVKNO3 -1
#define ind_PIPN -1
#define ind_TC4H9NO3 -1
#define ind_EPXDLPAN -1
#define ind_MALDIALPAN -1
#define ind_NBZFUO2 -1
#define ind_NBZFUONE -1
#define ind_NBZFUOOH -1
#define ind_NC4DCO2H -1
#define ind_LBUT1ENO2 -1
#define ind_LBUT1ENOOH -1
#define ind_LHMVKABO2 -1
#define ind_LHMVKABOOH -1
#define ind_LBUT1ENNO3 -1
#define ind_LMEKNO3 -1
#define ind_C1ODC2O2C4OD -1
#define ind_C1ODC2O2C4OOH -1
#define ind_C1ODC2OOHC4OD -1
#define ind_C1ODC3O2C4OOH -1
#define ind_C1OOHC2O2C4OD -1
#define ind_C1OOHC2OOHC4OD -1
#define ind_C1OOHC3O2C4OD -1
#define ind_C4MDIAL -1
#define ind_C511O2 -1
#define ind_C511OOH -1
#define ind_C512O2 -1
#define ind_C512OOH -1
#define ind_C513CO -1
#define ind_C513O2 -1
#define ind_C513OOH -1
#define ind_C514O2 -1
#define ind_C514OOH -1
#define ind_C59O2 -1
#define ind_C59OOH -1
#define ind_CHOC3COCO3 -1
#define ind_CHOC3COOOH -1
#define ind_CO13C4CHO -1
#define ind_CO23C4CHO -1
#define ind_CO23C4CO3 -1
#define ind_CO23C4CO3H -1
#define ind_DB1O -1
#define ind_DB1O2 -1
#define ind_DB1OOH -1
#define ind_DB2O2 -1
#define ind_DB2OOH -1
#define ind_HCOC5 -1
#define ind_ISOPAB -1
#define ind_ISOPAOH -1
#define ind_ISOPBO2 -1
#define ind_ISOPBOH -1
#define ind_ISOPBOOH -1
#define ind_ISOPCD -1
#define ind_ISOPDO2 -1
#define ind_ISOPDOH -1
#define ind_ISOPDOOH -1
#define ind_MBO -1
#define ind_MBOACO -1
#define ind_MBOCOCO -1
#define ind_ME3FURAN -1
#define ind_ZCO3C23DBCOD -1
#define ind_ZCODC23DBCOOH -1
#define ind_ACCOMECHO -1
#define ind_ACCOMECO3 -1
#define ind_ACCOMECO3H -1
#define ind_C24O3CCO2H -1
#define ind_C4CO2DBCO3 -1
#define ind_C4CO2DCO3H -1
#define ind_C5134CO2OH -1
#define ind_C54CO -1
#define ind_C5CO14O2 -1
#define ind_C5CO14OH -1
#define ind_C5CO14OOH -1
#define ind_C5DIALCO -1
#define ind_C5DIALO2 -1
#define ind_C5DIALOOH -1
#define ind_C5DICARB -1
#define ind_C5DICARBO2 -1
#define ind_C5DICAROOH -1
#define ind_MC3ODBCO2H -1
#define ind_MMALANHY -1
#define ind_MMALANHYO2 -1
#define ind_MMALNHYOOH -1
#define ind_TLFUO2 -1
#define ind_TLFUONE -1
#define ind_TLFUOOH -1
#define ind_C4MCONO3OH -1
#define ind_C514NO3 -1
#define ind_C5PAN9 -1
#define ind_CHOC3COPAN -1
#define ind_DB1NO3 -1
#define ind_ISOPBDNO3O2 -1
#define ind_ISOPBNO3 -1
#define ind_ISOPDNO3 -1
#define ind_NC4CHO -1
#define ind_NC4OHCO3 -1
#define ind_NC4OHCO3H -1
#define ind_NC4OHCPAN -1
#define ind_NISOPO2 -1
#define ind_NISOPOOH -1
#define ind_NMBOBCO -1
#define ind_ZCPANC23DBCOD -1
#define ind_ACCOMEPAN -1
#define ind_C4CO2DBPAN -1
#define ind_C5COO2NO2 -1
#define ind_NC4MDCO2H -1
#define ind_NTLFUO2 -1
#define ind_NTLFUOOH -1
#define ind_LC578O2 -1
#define ind_LC578OOH -1
#define ind_LDISOPACO -1
#define ind_LDISOPACO2 -1
#define ind_LHC4ACCHO -1
#define ind_LHC4ACCO2H -1
#define ind_LHC4ACCO3 -1
#define ind_LHC4ACCO3H -1
#define ind_LIEPOX -1
#define ind_LISOPACO -1
#define ind_LISOPACO2 -1
#define ind_LISOPACOOH -1
#define ind_LISOPEFO -1
#define ind_LISOPEFO2 -1
#define ind_LMBOABO2 -1
#define ind_LMBOABOOH -1
#define ind_LME3FURANO2 -1
#define ind_LZCO3HC23DBCOD -1
#define ind_LC5PAN1719 -1
#define ind_LISOPACNO3 -1
#define ind_LISOPACNO3O2 -1
#define ind_LMBOABNO3 -1
#define ind_LNISO3 -1
#define ind_LNISOOH -1
#define ind_LNMBOABO2 -1
#define ind_LNMBOABOOH -1
#define ind_C614CO -1
#define ind_C614O2 -1
#define ind_C614OOH -1
#define ind_CO235C5CHO -1
#define ind_CO235C6O2 -1
#define ind_CO235C6OOH -1
#define ind_BENZENE -1
#define ind_BZBIPERO2 -1
#define ind_BZBIPEROOH -1
#define ind_BZEMUCCO -1
#define ind_BZEMUCCO2H -1
#define ind_BZEMUCCO3 -1
#define ind_BZEMUCCO3H -1
#define ind_BZEMUCO2 -1
#define ind_BZEMUCOOH -1
#define ind_BZEPOXMUC -1
#define ind_BZOBIPEROH -1
#define ind_C5CO2DBCO3 -1
#define ind_C5CO2DCO3H -1
#define ind_C5CO2OHCO3 -1
#define ind_C5COOHCO3H -1
#define ind_C6125CO -1
#define ind_C615CO2O2 -1
#define ind_C615CO2OOH -1
#define ind_C6CO4DB -1
#define ind_C6H5O -1
#define ind_C6H5O2 -1
#define ind_C6H5OOH -1
#define ind_CATEC1O -1
#define ind_CATEC1O2 -1
#define ind_CATEC1OOH -1
#define ind_CATECHOL -1
#define ind_CPDKETENE -1
#define ind_PBZQCO -1
#define ind_PBZQO2 -1
#define ind_PBZQONE -1
#define ind_PBZQOOH -1
#define ind_PHENO2 -1
#define ind_PHENOL -1
#define ind_PHENOOH -1
#define ind_C614NO3 -1
#define ind_BZBIPERNO3 -1
#define ind_BZEMUCNO3 -1
#define ind_BZEMUCPAN -1
#define ind_C5CO2DBPAN -1
#define ind_C5CO2OHPAN -1
#define ind_DNPHEN -1
#define ind_DNPHENO2 -1
#define ind_DNPHENOOH -1
#define ind_HOC6H4NO2 -1
#define ind_NBZQO2 -1
#define ind_NBZQOOH -1
#define ind_NCATECHOL -1
#define ind_NCATECO2 -1
#define ind_NCATECOOH -1
#define ind_NCPDKETENE -1
#define ind_NDNPHENO2 -1
#define ind_NDNPHENOOH -1
#define ind_NNCATECO2 -1
#define ind_NNCATECOOH -1
#define ind_NPHEN1O -1
#define ind_NPHEN1O2 -1
#define ind_NPHEN1OOH -1
#define ind_NPHENO2 -1
#define ind_NPHENOOH -1
#define ind_C235C6CO3H -1
#define ind_C716O2 -1
#define ind_C716OOH -1
#define ind_C721O2 -1
#define ind_C721OOH -1
#define ind_C722O2 -1
#define ind_C722OOH -1
#define ind_CO235C6CHO -1
#define ind_CO235C6CO3 -1
#define ind_MCPDKETENE -1
#define ind_ROO6R3O -1
#define ind_ROO6R3O2 -1
#define ind_ROO6R5O2 -1
#define ind_BENZAL -1
#define ind_C6CO2OHCO3 -1
#define ind_C6COOHCO3H -1
#define ind_C6H5CH2O2 -1
#define ind_C6H5CH2OOH -1
#define ind_C6H5CO3 -1
#define ind_C6H5CO3H -1
#define ind_C7CO4DB -1
#define ind_CRESO2 -1
#define ind_CRESOL -1
#define ind_CRESOOH -1
#define ind_MCATEC1O -1
#define ind_MCATEC1O2 -1
#define ind_MCATEC1OOH -1
#define ind_MCATECHOL -1
#define ind_OXYL1O2 -1
#define ind_OXYL1OOH -1
#define ind_PHCOOH -1
#define ind_PTLQCO -1
#define ind_PTLQO2 -1
#define ind_PTLQONE -1
#define ind_PTLQOOH -1
#define ind_TLBIPERO2 -1
#define ind_TLBIPEROOH -1
#define ind_TLEMUCCO -1
#define ind_TLEMUCCO2H -1
#define ind_TLEMUCCO3 -1
#define ind_TLEMUCCO3H -1
#define ind_TLEMUCO2 -1
#define ind_TLEMUCOOH -1
#define ind_TLEPOXMUC -1
#define ind_TLOBIPEROH -1
#define ind_TOL1O -1
#define ind_TOLUENE -1
#define ind_C7PAN3 -1
#define ind_C6CO2OHPAN -1
#define ind_C6H5CH2NO3 -1
#define ind_DNCRES -1
#define ind_DNCRESO2 -1
#define ind_DNCRESOOH -1
#define ind_MNCATECH -1
#define ind_MNCATECO2 -1
#define ind_MNCATECOOH -1
#define ind_MNCPDKETENE -1
#define ind_MNNCATCOOH -1
#define ind_MNNCATECO2 -1
#define ind_NCRES1O -1
#define ind_NCRES1O2 -1
#define ind_NCRES1OOH -1
#define ind_NCRESO2 -1
#define ind_NCRESOOH -1
#define ind_NDNCRESO2 -1
#define ind_NDNCRESOOH -1
#define ind_NPTLQO2 -1
#define ind_NPTLQOOH -1
#define ind_PBZN -1
#define ind_TLBIPERNO3 -1
#define ind_TLEMUCNO3 -1
#define ind_TLEMUCPAN -1
#define ind_TOL1OHNO2 -1
#define ind_C721CHO -1
#define ind_C721CO3 -1
#define ind_C721CO3H -1
#define ind_C810O2 -1
#define ind_C810OOH -1
#define ind_C811O2 -1
#define ind_C812O2 -1
#define ind_C812OOH -1
#define ind_C813O2 -1
#define ind_C813OOH -1
#define ind_C85O2 -1
#define ind_C85OOH -1
#define ind_C86O2 -1
#define ind_C86OOH -1
#define ind_C89O2 -1
#define ind_C89OOH -1
#define ind_C8BC -1
#define ind_C8BCCO -1
#define ind_C8BCO2 -1
#define ind_C8BCOOH -1
#define ind_NORPINIC -1
#define ind_EBENZ -1
#define ind_LXYL -1
#define ind_STYRENE -1
#define ind_STYRENO2 -1
#define ind_STYRENOOH -1
#define ind_C721PAN -1
#define ind_C810NO3 -1
#define ind_C89NO3 -1
#define ind_C8BCNO3 -1
#define ind_NSTYRENO2 -1
#define ind_NSTYRENOOH -1
#define ind_C811CO3 -1
#define ind_C811CO3H -1
#define ind_C85CO3 -1
#define ind_C85CO3H -1
#define ind_C89CO2H -1
#define ind_C89CO3 -1
#define ind_C89CO3H -1
#define ind_C96O2 -1
#define ind_C96OOH -1
#define ind_C97O2 -1
#define ind_C97OOH -1
#define ind_C98O2 -1
#define ind_C98OOH -1
#define ind_NOPINDCO -1
#define ind_NOPINDO2 -1
#define ind_NOPINDOOH -1
#define ind_NOPINONE -1
#define ind_NOPINOO -1
#define ind_NORPINAL -1
#define ind_NORPINENOL -1
#define ind_PINIC -1
#define ind_RO6R3P -1
#define ind_C811PAN -1
#define ind_C89PAN -1
#define ind_C96NO3 -1
#define ind_C9PAN2 -1
#define ind_LTMB -1
#define ind_APINAOO -1
#define ind_APINBOO -1
#define ind_APINENE -1
#define ind_BPINAO2 -1
#define ind_BPINAOOH -1
#define ind_BPINENE -1
#define ind_C106O2 -1
#define ind_C106OOH -1
#define ind_C109CO -1
#define ind_C109O2 -1
#define ind_C109OOH -1
#define ind_C96CO3 -1
#define ind_CAMPHENE -1
#define ind_CARENE -1
#define ind_MENTHEN6ONE -1
#define ind_OH2MENTHEN6ONE -1
#define ind_OHMENTHEN6ONEO2 -1
#define ind_PERPINONIC -1
#define ind_PINAL -1
#define ind_PINALO2 -1
#define ind_PINALOOH -1
#define ind_PINENOL -1
#define ind_PINONIC -1
#define ind_RO6R1O2 -1
#define ind_RO6R3O2 -1
#define ind_RO6R3OOH -1
#define ind_ROO6R1O2 -1
#define ind_SABINENE -1
#define ind_BPINANO3 -1
#define ind_C106NO3 -1
#define ind_C10PAN2 -1
#define ind_PINALNO3 -1
#define ind_RO6R1NO3 -1
#define ind_RO6R3NO3 -1
#define ind_ROO6R1NO3 -1
#define ind_LAPINABNO3 -1
#define ind_LAPINABO2 -1
#define ind_LAPINABOOH -1
#define ind_LNAPINABO2 -1
#define ind_LNAPINABOOH -1
#define ind_LNBPINABO2 -1
#define ind_LNBPINABOOH -1
#define ind_LHAROM -1
#define ind_CHF3 -1
#define ind_CHF2CF3 -1
#define ind_CH3CF3 -1
#define ind_CH2F2 -1
#define ind_CH3CHF2 -1
#define ind_CF2ClCF2Cl -1
#define ind_CF2ClCFCl2 -1
#define ind_CF3CF2Cl -1
#define ind_CH2Cl2 -1
#define ind_CH2FCF3 -1
#define ind_CH3CFCl2 -1
#define ind_CHCl3 -1
#define ind_CHF2Cl -1
#define ind_LBROMINE -1
#define ind_C3H7I -1
#define ind_CH2ClI -1
#define ind_CH2I2 -1
#define ind_HI -1
#define ind_HIO3 -1
#define ind_HOI -1
#define ind_I -1
#define ind_I2 -1
#define ind_I2O2 -1
#define ind_IBr -1
#define ind_ICl -1
#define ind_INO2 -1
#define ind_INO3 -1
#define ind_IO -1
#define ind_IPART -1
#define ind_OIO -1
#define ind_OCS -1
#define ind_S -1
#define ind_SF6 -1
#define ind_SH -1
#define ind_SO -1
#define ind_SO3 -1
#define ind_LSULFUR -1
#define ind_Hg -1
#define ind_HgO -1
#define ind_HgCl -1
#define ind_HgCl2 -1
#define ind_HgBr -1
#define ind_HgBr2 -1
#define ind_ClHgBr -1
#define ind_BrHgOBr -1
#define ind_ClHgOBr -1
#define ind_RGM_cs -1
#define ind_PRODUCTS -1
#define ind_M -1
#define ind_Op -1
#define ind_O2p -1
#define ind_Np -1
#define ind_N2p -1
#define ind_NOp -1
#define ind_em -1
#define ind_kJmol -1
#define ind_O4Sp -1
#define ind_O2Dp -1
#define ind_O2Pp -1
#define ind_LTERP -1
#define ind_LALK4 -1
#define ind_LALK5 -1
#define ind_LARO1 -1
#define ind_LARO2 -1
#define ind_LOLE1 -1
#define ind_LOLE2 -1
#define ind_LfPOG02 -1
#define ind_LfPOG03 -1
#define ind_LfPOG04 -1
#define ind_LfPOG05 -1
#define ind_LbbPOG02 -1
#define ind_LbbPOG03 -1
#define ind_LbbPOG04 -1
#define ind_LfSOGsv01 -1
#define ind_LfSOGsv02 -1
#define ind_LbbSOGsv01 -1
#define ind_LbbSOGsv02 -1
#define ind_LfSOGiv01 -1
#define ind_LfSOGiv02 -1
#define ind_LfSOGiv03 -1
#define ind_LfSOGiv04 -1
#define ind_LbbSOGiv01 -1
#define ind_LbbSOGiv02 -1
#define ind_LbbSOGiv03 -1
#define ind_LbSOGv01 -1
#define ind_LbSOGv02 -1
#define ind_LbSOGv03 -1
#define ind_LbSOGv04 -1
#define ind_LbOSOGv01 -1
#define ind_LbOSOGv02 -1
#define ind_LbOSOGv03 -1
#define ind_LaSOGv01 -1
#define ind_LaSOGv02 -1
#define ind_LaSOGv03 -1
#define ind_LaSOGv04 -1
#define ind_LaOSOGv01 -1
#define ind_LaOSOGv02 -1
#define ind_LaOSOGv03 -1
#define ind_ACBZO2 -1
#define ind_ALKNO3 -1
#define ind_ALKO2 -1
#define ind_ALKOH -1
#define ind_ALKOOH -1
#define ind_BCARY -1
#define ind_BENZO2 -1
#define ind_BENZOOH -1
#define ind_BEPOMUC -1
#define ind_BIGALD1 -1
#define ind_BIGALD2 -1
#define ind_BIGALD3 -1
#define ind_BIGALD4 -1
#define ind_BIGALKANE -1
#define ind_BIGENE -1
#define ind_BrONO -1
#define ind_BZALD -1
#define ind_BZOO -1
#define ind_BZOOH -1
#define ind_C3H7O2 -1
#define ind_C3H7OOH -1
#define ind_CFC113 -1
#define ind_CFC114 -1
#define ind_CFC115 -1
#define ind_COF2 -1
#define ind_COFCL -1
#define ind_DICARBO2 -1
#define ind_ELVOC -1
#define ind_ENEO2 -1
#define ind_EOOH -1
#define ind_F -1
#define ind_H1202 -1
#define ind_H2402 -1
#define ind_HCFC141B -1
#define ind_HCFC142B -1
#define ind_HCFC22 -1
#define ind_HF -1
#define ind_HOCH2OO -1
#define ind_HPALD -1
#define ind_IEC1O2 -1
#define ind_LIECHO -1
#define ind_LIECO3 -1
#define ind_LIECO3H -1
#define ind_LIMON -1
#define ind_LISOPNO3NO3 -1
#define ind_LISOPNO3O2 -1
#define ind_LISOPNO3OOH -1
#define ind_LISOPOOHO2 -1
#define ind_LISOPOOHOOH -1
#define ind_MALO2 -1
#define ind_MBONO3O2 -1
#define ind_MBOO2 -1
#define ind_MBOOOH -1
#define ind_MDIALO2 -1
#define ind_MEKNO3 -1
#define ind_MVKN -1
#define ind_MYRC -1
#define ind_NTERPNO3 -1
#define ind_NTERPO2 -1
#define ind_PACALD -1
#define ind_PBZNIT -1
#define ind_TEPOMUC -1
#define ind_TERP2O2 -1
#define ind_TERP2OOH -1
#define ind_TERPNO3 -1
#define ind_TERPO2 -1
#define ind_TERPOOH -1
#define ind_TERPROD1 -1
#define ind_TERPROD2 -1
#define ind_TOLO2 -1
#define ind_TOLOOH -1
#define ind_XYLENO2 -1
#define ind_XYLENOOH -1
#define ind_XYLOL -1
#define ind_XYLOLO2 -1
#define ind_XYLOLOOH -1
#define ind_O2_1D -1
#define ind_O2_1S -1
#define ind_ONIT -1
#define ind_C4H8 -1
#define ind_C4H9O3 -1
#define ind_C5H12 -1
#define ind_C5H11O2 -1
#define ind_C5H6O2 -1
#define ind_HYDRALD -1
#define ind_ISOPO2 -1
#define ind_C5H9O3 -1
#define ind_ISOPOOH -1
#define ind_C5H12O2 -1
#define ind_ONITR -1
#define ind_C5H10O4 -1
#define ind_ROO6R5P -1
#define ind_NH4 -1
#define ind_SO4 -1
#define ind_HCO -1
#define ind_ISPD -1
#define ind_ClOO -1
#define ind_Rn -1
#define ind_Pb -1
#define ind_XO2 -1
#define ind_XO2N -1
#define ind_ROOH -1
#define ind_OLE -1
#define ind_ROR -1
#define ind_ORGNTR -1
#define ind_ACO2 -1
#define ind_PAR -1
#define ind_RXPAR -1
#define ind_OHv0 -1
#define ind_OHv1 -1
#define ind_OHv2 -1
#define ind_OHv3 -1
#define ind_OHv4 -1
#define ind_OHv5 -1
#define ind_OHv6 -1
#define ind_OHv7 -1
#define ind_OHv8 -1
#define ind_OHv9 -1
#define ind_O1S -1
#define ind_O21d -1
#define ind_O2b1s -1
#define ind_O2c1s -1
#define ind_O2x -1
#define ind_O2A3D -1
#define ind_O2A3S -1
#define ind_O25P -1
#define ind_O2_a01 -1
#define ind_O3_a01 -1
#define ind_OH_a01 -1
#define ind_HO2_a01 -1
#define ind_H2O_a01 -1
#define ind_H2O2_a01 -1
#define ind_NH3_a01 -1
#define ind_NO_a01 -1
#define ind_NO2_a01 -1
#define ind_NO3_a01 -1
#define ind_HONO_a01 -1
#define ind_HNO3_a01 -1
#define ind_HNO4_a01 -1
#define ind_CH3OH_a01 -1
#define ind_HCOOH_a01 -1
#define ind_HCHO_a01 -1
#define ind_CH3O2_a01 -1
#define ind_CH3OOH_a01 -1
#define ind_CO2_a01 -1
#define ind_CH3CO2H_a01 -1
#define ind_PAN_a01 -1
#define ind_CH3CHO_a01 -1
#define ind_CH3COCH3_a01 -1
#define ind_Cl_a01 -1
#define ind_Cl2_a01 -1
#define ind_HCl_a01 -1
#define ind_HOCl_a01 -1
#define ind_Br_a01 -1
#define ind_Br2_a01 -1
#define ind_HBr_a01 -1
#define ind_HOBr_a01 -1
#define ind_BrCl_a01 -1
#define ind_I2_a01 -1
#define ind_IO_a01 -1
#define ind_HOI_a01 -1
#define ind_ICl_a01 -1
#define ind_IBr_a01 -1
#define ind_SO2_a01 -1
#define ind_H2SO4_a01 -1
#define ind_DMS_a01 -1
#define ind_DMSO_a01 -1
#define ind_Hg_a01 -1
#define ind_HgO_a01 -1
#define ind_HgOHOH_a01 -1
#define ind_HgOHCl_a01 -1
#define ind_HgCl2_a01 -1
#define ind_HgBr2_a01 -1
#define ind_HgSO3_a01 -1
#define ind_ClHgBr_a01 -1
#define ind_BrHgOBr_a01 -1
#define ind_ClHgOBr_a01 -1
#define ind_FeOH3_a01 -1
#define ind_FeCl3_a01 -1
#define ind_FeF3_a01 -1
#define ind_O2m_a01 -1
#define ind_OHm_a01 -1
#define ind_HO2m_a01 -1
#define ind_O2mm_a01 -1
#define ind_Hp_a01 -1
#define ind_NH4p_a01 -1
#define ind_NO2m_a01 -1
#define ind_NO3m_a01 -1
#define ind_NO4m_a01 -1
#define ind_CO3m_a01 -1
#define ind_HCOOm_a01 -1
#define ind_HCO3m_a01 -1
#define ind_CH3COOm_a01 -1
#define ind_Clm_a01 -1
#define ind_Cl2m_a01 -1
#define ind_ClOm_a01 -1
#define ind_ClOHm_a01 -1
#define ind_Brm_a01 -1
#define ind_Br2m_a01 -1
#define ind_BrOm_a01 -1
#define ind_BrOHm_a01 -1
#define ind_BrCl2m_a01 -1
#define ind_Br2Clm_a01 -1
#define ind_Im_a01 -1
#define ind_IO2m_a01 -1
#define ind_IO3m_a01 -1
#define ind_ICl2m_a01 -1
#define ind_IBr2m_a01 -1
#define ind_SO3m_a01 -1
#define ind_SO3mm_a01 -1
#define ind_SO4m_a01 -1
#define ind_SO4mm_a01 -1
#define ind_SO5m_a01 -1
#define ind_HSO3m_a01 -1
#define ind_HSO4m_a01 -1
#define ind_HSO5m_a01 -1
#define ind_CH3SO3m_a01 -1
#define ind_CH2OHSO3m_a01 -1
#define ind_Hgp_a01 -1
#define ind_Hgpp_a01 -1
#define ind_HgOHp_a01 -1
#define ind_HgClp_a01 -1
#define ind_HgBrp_a01 -1
#define ind_HgSO32mm_a01 -1
#define ind_Fepp_a01 -1
#define ind_FeOpp_a01 -1
#define ind_FeOHp_a01 -1
#define ind_FeOH2p_a01 -1
#define ind_FeClp_a01 -1
#define ind_Feppp_a01 -1
#define ind_FeHOpp_a01 -1
#define ind_FeHO2pp_a01 -1
#define ind_FeOHpp_a01 -1
#define ind_FeOH4m_a01 -1
#define ind_FeOHHO2p_a01 -1
#define ind_FeClpp_a01 -1
#define ind_FeCl2p_a01 -1
#define ind_FeBrpp_a01 -1
#define ind_FeBr2p_a01 -1
#define ind_FeFpp_a01 -1
#define ind_FeF2p_a01 -1
#define ind_FeSO3p_a01 -1
#define ind_FeSO4p_a01 -1
#define ind_FeSO42m_a01 -1
#define ind_FeOH2Fepppp_a01 -1
#define ind_D1O_a01 -1
#define ind_Nap_a01 -1
#define ind_LossO3Su -1

#define ihs_N2O5_H2O 0
#define ihs_HOCl_HCl 1
#define ihs_ClNO3_HCl 2
#define ihs_ClNO3_H2O 3
#define ihs_N2O5_HCl 4
#define ihs_ClNO3_HBr 5
#define ihs_BrNO3_HCl 6
#define ihs_HOCl_HBr 7
#define ihs_HOBr_HCl 8
#define ihs_HOBr_HBr 9
#define ihs_BrNO3_H2O 10
#define ihs_Hg 11
#define ihs_RGM 12

#define iht_N2O5 0
#define iht_HNO3 1
#define iht_Hg 2
#define iht_RGM 3

#define k_C6H5O_NO2    (2.08E-12)
#define k_C6H5O_O3     (2.86E-13)
#define k_adsecprim   (3.0E-11)
#define k_adtertprim   (5.7E-11 )
#define f_soh   (3.44)
#define f_toh   (2.68)
#define f_sooh   (7.)
#define f_tooh   (7.)
#define f_ono2   (0.04 )
#define f_ch2ono2   (0.2)
#define f_cpan  (.25)
#define f_allyl   (3.6)
#define f_alk  (1.23)
#define f_cho   (0.55)
#define f_co2h   (1.67)
#define f_co   (0.73)
#define f_o   (8.15)
#define f_pch2oh   (1.29)
#define f_tch2oh   (0.53)
#define a_pan   (0.56      )
#define a_cho   (0.31   )
#define a_coch3   (0.76 )
#define a_ch2ono2   (0.64  )
#define a_ch2oh   (1.7  )
#define a_ch2ooh   (1.7 )
#define a_coh   (2.2       )
#define a_cooh   (2.2   )
#define a_co2h   (0.25)

#define ifun 0
#define ijac 1
#define istp 2
#define iacc 3
#define irej 4
#define idec 5
#define isol 6
#define isng 7
#define itexit 0
#define ihexit 1

#define ZERO 0.0
#define ONE 1.0
#define HALF 0.5


/*
 * Fortran to C macros 
 * GPU-friendly array deffinition 
 * i:VL_GLO, j:NVAR 
 *
 */
#define conc(i,j)    conc[(j)*VL_GLO+(i)]
#define khet_st(i,j) khet_st[(j)*VL_GLO+(i)]
#define khet_tr(i,j) khet_tr[(j)*VL_GLO+(i)]
#define jx(i,j)      jx[j*VL_GLO+i]
#define istatus(i,j) istatus[(j)*(VL_GLO)+(i)]   
#define rstatus(i,j) rstatus[(j)*(VL_GLO)+(i)]


#define ROUND128(X)  (X + (128 - 1)) & ~(128 - 1)

#define rconst(i,j)  rconst[(j)]


/* Temporary arrays allocated in stack */
#define var(i,j)     var[(j)]
#define fix(i,j)     fix[(j)]
#define jcb(i,j)     jcb[(j)]
#define varDot(i,j)  varDot[j]
#define varNew(i,j) varNew[(j)]
#define Fcn0(i,j)   Fcn0[(j)]
#define Fcn(i,j)    Fcn[(j)]
#define Fcn(i,j)    Fcn[(j)]
#define dFdT(i,j)   dFdT[(j)]
#define varErr(i,j) varErr[(j)]
#define K(i,j,k) K[(j)*(NVAR)+(k)]
#define jac0(i,j)    jac0[(j)]    
#define Ghimj(i,j)   Ghimj[(j)]   


/* Enable debug flags for GPU */
//#define DEBUG

#ifdef DEBUG
#define GPU_DEBUG()\
    gpuErrchk( cudaPeekAtLastError()   ); \
    gpuErrchk( cudaDeviceSynchronize() ); 

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

static inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#else 
/* If debug flags are disabled */
#define GPU_DEBUG()
#define gpuErrchk(ans) ans
#endif

/** prefetches into L1 cache */
__device__ inline void prefetch_gl1(const void *p) {
#if __CUDA_ARCH__ <= 300
        asm("prefetch.global.L1 [%0];": :"l"(p));
#endif
}
__device__ inline void prefetch_ll1(const void *p) {
#if __CUDA_ARCH__ <= 300
        asm("prefetch.local.L1 [%0];": :"l"(p));
#endif
}

/** prefetches into L2 cache */
__device__ inline void prefetch_gl2(const void *p) {
#if __CUDA_ARCH__ <= 300
        asm("prefetch.global.L2 [%0];": :"l"(p));
#endif
}
__device__ inline void prefetch_ll2(const void *p) {
#if __CUDA_ARCH__ <= 300
        asm("prefetch.local.L2 [%0];": :"l"(p));
#endif
}



__device__ void  update_rconst(const double * __restrict__ var,
			       const double * __restrict__ khet_st, const double * __restrict__ khet_tr,
			       const double * __restrict__ jx, double * __restrict__ rconst,
 			       const double * __restrict__ temp_gpu,
 			       const double * __restrict__ press_gpu,
 			       const double * __restrict__ cair_gpu,
			       const int VL_GLO);

/* This runs on CPU */
double machine_eps_flt()
{
    double machEps = 1.0f;

    do
    {
        machEps /= 2.0f;
        // If next epsilon yields 1, then break, because current
        // epsilon is the machine epsilon.
    }
    while ((double)(1.0 + (machEps/2.0)) != 1.0);

    return machEps;
}

/* This runs on GPU */
__device__ double machine_eps_flt_cuda() 
{
    typedef union 
    {
        long  i64;
        double f64;
    } flt_64;

    flt_64 s;

    s.f64 = 1.;
    s.i64++;
    return (s.f64 - 1.);
}

__device__  static double alpha_AN(const int n, const int ro2type, const double temp, const double cair){
    double alpha=2.E-22, beta=1.0, Yinf_298K=0.43,  F=0.41, m0=0., minf=8.0;
    double Y0_298K, Y0_298K_tp, Yinf_298K_t, zeta, k_ratio, alpha_a;
    /*  IF (ro2type = 1) THEN   m = 0.4                !   primary RO2
        ELSE IF (ro2type = 2) THEN  m = 1.                 !   secondary RO2
        ELSE IF (ro2type = 3) THEN  m = 0.3                !   tertiary RO2
        ELSE  m = 1.
  */
    double m = 1.;
    Y0_298K     = alpha*exp(beta*n);
    Y0_298K_tp  = Y0_298K *cair *pow((temp/298.),(- m0));
    Yinf_298K_t = Yinf_298K * pow((temp/298.),(- minf));
    zeta        = 1/(1+ pow(log10(Y0_298K_tp/Yinf_298K_t),2));
    k_ratio     = (Y0_298K_tp/(1+ Y0_298K_tp/Yinf_298K_t))*pow(F,zeta);
    alpha_a    = k_ratio/(1+ k_ratio) *m;
    return alpha_a;
}
__device__  static double alpha_AN(const int n, const int ro2type, const int bcarb, const int gcarb, const int abic, const double temp, const double cair){
    double alpha=2.E-22, beta=1.0, Yinf_298K=0.43,  F=0.41, m0=0., minf=8.0;
    double Y0_298K, Y0_298K_tp, Yinf_298K_t, zeta, k_ratio, alpha_a;
    double bcf=1., gcf=1., abf=1.;
    double m = 1.; //According to Teng, ref3189

if (bcarb == 1) { bcf = 0.19; }// derived from Praske, ref3190: alpha_AN = 0.03 for the secondary HMKO2 relative to alpha_AN for 6C RO2 (0.16)
if (gcarb == 1) {gcf = 0.44; }// derived from Praske, ref3190: alpha_AN = 0.07 for the primary HMKO2 relative to alpha_AN for 6C RO2 (0.16)
if (abic == 1) { abf = 0.24; }// derived from the ratio of AN- yield for toluene from Elrod et al. (ref3180), 5.5 0x1.9206e69676542p+ 229t & 
                              // 200 torr, and this SAR for linear alkyl RO2 with 9 heavy atoms, 23.3%

    Y0_298K     = alpha*exp(beta*n);
    Y0_298K_tp  = Y0_298K *cair *pow((temp/298.),(- m0));
    Yinf_298K_t = Yinf_298K * pow((temp/298.),(- minf));
    zeta        = 1/(1+ pow(log10(Y0_298K_tp/Yinf_298K_t),2));
    k_ratio     = (Y0_298K_tp/(1+ Y0_298K_tp/Yinf_298K_t))*pow(F,zeta);
    alpha_a    = k_ratio/(1+ k_ratio) *m*bcf*gcf*abf;
    return alpha_a;
}
__device__  static double k_RO2_HO2(const double temp, const int nC){
    return 2.91e-13*exp(1300./temp)*(1.-exp(-0.245*nC)); // ref1630
}
__device__ double ros_ErrorNorm(double * __restrict__ var, double * __restrict__ varNew, double * __restrict__ varErr, 
                                const double * __restrict__ absTol, const double * __restrict__ relTol,
                                const int vectorTol )
{
    double err, scale, varMax;


    err = ZERO;

    if (vectorTol){
        for (int i=0;i<NVAR - 16;i+=16){
            prefetch_ll1(&varErr[i]);
            prefetch_ll1(&absTol[i]);
            prefetch_ll1(&relTol[i]);
            prefetch_ll1(&var[i]);
            prefetch_ll1(&varNew[i]);
        }

        for (int i=0; i<NVAR; i++)
        {
            varMax = fmax(fabs(var[i]),fabs(varNew[i]));
            scale = absTol[i]+ relTol[i]*varMax;

            err += pow((double)varErr[i]/scale,2.0);
        }
        err  = sqrt((double) err/NVAR);
    }else{
        for (int i=0;i<NVAR - 16;i+=16){
            prefetch_ll1(&varErr[i]);
            prefetch_ll1(&var[i]);
            prefetch_ll1(&varNew[i]);
        }

        for (int i=0; i<NVAR; i++)
        {
            varMax = fmax(fabs(var[i]),fabs(varNew[i]));

            scale = absTol[0]+ relTol[0]*varMax;
            err += pow((double)varErr[i]/scale,2.0);
        }
        err  = sqrt((double) err/NVAR);
    }

    return err;


}

__device__ void kppSolve(const double * __restrict__ Ghimj, double * __restrict__ K, 
                         const int istage, const int ros_S ){
    int index = blockIdx.x*blockDim.x+threadIdx.x;

       K = &K[istage*NVAR];

        K[7] = K[7]- Ghimj[7]*K[1]- Ghimj[8]*K[2];
        K[8] = K[8]- Ghimj[23]*K[1]- Ghimj[24]*K[2];
        K[14] = K[14]- Ghimj[50]*K[5]- Ghimj[51]*K[6];
        K[19] = K[19]- Ghimj[67]*K[4];
        K[31] = K[31]- Ghimj[188]*K[1]- Ghimj[189]*K[2];
        K[32] = K[32]- Ghimj[193]*K[1];
        K[34] = K[34]- Ghimj[205]*K[0];
        K[60] = K[60]- Ghimj[309]*K[59];
        K[70] = K[70]- Ghimj[351]*K[61];
        K[85] = K[85]- Ghimj[426]*K[79];
        K[86] = K[86]- Ghimj[434]*K[62]- Ghimj[435]*K[69];
        K[87] = K[87]- Ghimj[442]*K[70]- Ghimj[443]*K[84];
        K[90] = K[90]- Ghimj[468]*K[80];
        K[92] = K[92]- Ghimj[487]*K[47]- Ghimj[488]*K[84];
        K[93] = K[93]- Ghimj[495]*K[49]- Ghimj[496]*K[69];
        K[94] = K[94]- Ghimj[502]*K[72]- Ghimj[503]*K[86]- Ghimj[504]*K[93];
        K[95] = K[95]- Ghimj[510]*K[58]- Ghimj[511]*K[77]- Ghimj[512]*K[82]- Ghimj[513]*K[91];
        K[96] = K[96]- Ghimj[535]*K[72]- Ghimj[536]*K[82]- Ghimj[537]*K[94];
        K[99] = K[99]- Ghimj[563]*K[68]- Ghimj[564]*K[85];
        K[100] = K[100]- Ghimj[572]*K[90];
        K[101] = K[101]- Ghimj[585]*K[83];
        K[102] = K[102]- Ghimj[598]*K[40]- Ghimj[599]*K[79];
        K[108] = K[108]- Ghimj[630]*K[64]- Ghimj[631]*K[67]- Ghimj[632]*K[82]- Ghimj[633]*K[91]- Ghimj[634]*K[94]- Ghimj[635]*K[106];
        K[109] = K[109]- Ghimj[647]*K[106];
        K[110] = K[110]- Ghimj[655]*K[66]- Ghimj[656]*K[91]- Ghimj[657]*K[106]- Ghimj[658]*K[109];
        K[111] = K[111]- Ghimj[666]*K[99]- Ghimj[667]*K[102]- Ghimj[668]*K[107];
        K[113] = K[113]- Ghimj[685]*K[64]- Ghimj[686]*K[82]- Ghimj[687]*K[106]- Ghimj[688]*K[110];
        K[115] = K[115]- Ghimj[703]*K[67]- Ghimj[704]*K[103]- Ghimj[705]*K[107];
        K[117] = K[117]- Ghimj[722]*K[48]- Ghimj[723]*K[49]- Ghimj[724]*K[71]- Ghimj[725]*K[79]- Ghimj[726]*K[85]- Ghimj[727]*K[102]- Ghimj[728]  *K[107]- Ghimj[729]*K[111]- Ghimj[730]*K[115];
        K[118] = K[118]- Ghimj[741]*K[100]- Ghimj[742]*K[105]- Ghimj[743]*K[112]- Ghimj[744]*K[116];
        K[119] = K[119]- Ghimj[758]*K[68]- Ghimj[759]*K[71]- Ghimj[760]*K[79]- Ghimj[761]*K[99]- Ghimj[762]*K[102]- Ghimj[763]*K[107]- Ghimj[764]  *K[111]- Ghimj[765]*K[115]- Ghimj[766]*K[117];
        K[120] = K[120]- Ghimj[777]*K[41]- Ghimj[778]*K[42]- Ghimj[779]*K[43]- Ghimj[780]*K[57]- Ghimj[781]*K[60]- Ghimj[782]*K[75]- Ghimj[783]  *K[92]- Ghimj[784]*K[97]- Ghimj[785]*K[98]- Ghimj[786]*K[107];
        K[121] = K[121]- Ghimj[798]*K[38]- Ghimj[799]*K[63]- Ghimj[800]*K[68]- Ghimj[801]*K[72]- Ghimj[802]*K[77]- Ghimj[803]*K[82]- Ghimj[804]  *K[85]- Ghimj[805]*K[86]- Ghimj[806]*K[93]- Ghimj[807]*K[94]- Ghimj[808]*K[96]- Ghimj[809]*K[99]- Ghimj[810]*K[102]- Ghimj[811] *K[106]- Ghimj[812]*K[107]- Ghimj[813]*K[108]- Ghimj[814]*K[109]- Ghimj[815]*K[110]- Ghimj[816]*K[111]- Ghimj[817]*K[113] - Ghimj[818]*K[115]- Ghimj[819]*K[117]- Ghimj[820]*K[119];
        K[122] = K[122]- Ghimj[831]*K[75]- Ghimj[832]*K[95]- Ghimj[833]*K[96]- Ghimj[834]*K[97]- Ghimj[835]*K[98]- Ghimj[836]*K[103]- Ghimj[837]  *K[106]- Ghimj[838]*K[107]- Ghimj[839]*K[108]- Ghimj[840]*K[109]- Ghimj[841]*K[110]- Ghimj[842]*K[113]- Ghimj[843]*K[115] - Ghimj[844]*K[119]- Ghimj[845]*K[120]- Ghimj[846]*K[121];
        K[123] = K[123]- Ghimj[861]*K[103]- Ghimj[862]*K[104]- Ghimj[863]*K[112]- Ghimj[864]*K[114]- Ghimj[865]*K[116]- Ghimj[866]*K[118]  - Ghimj[867]*K[119]- Ghimj[868]*K[121];
        K[124] = K[124]- Ghimj[885]*K[81]- Ghimj[886]*K[84]- Ghimj[887]*K[92]- Ghimj[888]*K[103]- Ghimj[889]*K[106]- Ghimj[890]*K[107]- Ghimj[891]  *K[110]- Ghimj[892]*K[114]- Ghimj[893]*K[120]- Ghimj[894]*K[121]- Ghimj[895]*K[122];
        K[125] = K[125]- Ghimj[910]*K[3]- Ghimj[911]*K[53]- Ghimj[912]*K[63]- Ghimj[913]*K[65]- Ghimj[914]*K[74]- Ghimj[915]*K[75]- Ghimj[916]  *K[81]- Ghimj[917]*K[86]- Ghimj[918]*K[93]- Ghimj[919]*K[94]- Ghimj[920]*K[98]- Ghimj[921]*K[102]- Ghimj[922]*K[104]- Ghimj[923] *K[106]- Ghimj[924]*K[107]- Ghimj[925]*K[109]- Ghimj[926]*K[113]- Ghimj[927]*K[114]- Ghimj[928]*K[117]- Ghimj[929]*K[119] - Ghimj[930]*K[120]- Ghimj[931]*K[121]- Ghimj[932]*K[122]- Ghimj[933]*K[124];
        K[126] = K[126]- Ghimj[948]*K[40]- Ghimj[949]*K[44]- Ghimj[950]*K[45]- Ghimj[951]*K[47]- Ghimj[952]*K[48]- Ghimj[953]*K[49]- Ghimj[954]  *K[52]- Ghimj[955]*K[53]- Ghimj[956]*K[54]- Ghimj[957]*K[55]- Ghimj[958]*K[56]- Ghimj[959]*K[57]- Ghimj[960]*K[58]- Ghimj[961] *K[61]- Ghimj[962]*K[62]- Ghimj[963]*K[63]- Ghimj[964]*K[64]- Ghimj[965]*K[65]- Ghimj[966]*K[66]- Ghimj[967]*K[67]- Ghimj[968] *K[68]- Ghimj[969]*K[69]- Ghimj[970]*K[70]- Ghimj[971]*K[71]- Ghimj[972]*K[72]- Ghimj[973]*K[73]- Ghimj[974]*K[74]- Ghimj[975] *K[75]- Ghimj[976]*K[76]- Ghimj[977]*K[77]- Ghimj[978]*K[78]- Ghimj[979]*K[79]- Ghimj[980]*K[81]- Ghimj[981]*K[82]- Ghimj[982] *K[84]- Ghimj[983]*K[85]- Ghimj[984]*K[86]- Ghimj[985]*K[87]- Ghimj[986]*K[88]- Ghimj[987]*K[89]- Ghimj[988]*K[91]- Ghimj[989] *K[92]- Ghimj[990]*K[93]- Ghimj[991]*K[94]- Ghimj[992]*K[95]- Ghimj[993]*K[96]- Ghimj[994]*K[97]- Ghimj[995]*K[98]- Ghimj[996] *K[99]- Ghimj[997]*K[100]- Ghimj[998]*K[101]- Ghimj[999]*K[102]- Ghimj[1000]*K[103]- Ghimj[1001]*K[104]- Ghimj[1002]*K[105] - Ghimj[1003]*K[106]- Ghimj[1004]*K[107]- Ghimj[1005]*K[108]- Ghimj[1006]*K[109]- Ghimj[1007]*K[110]- Ghimj[1008]*K[111] - Ghimj[1009]*K[112]- Ghimj[1010]*K[113]- Ghimj[1011]*K[114]- Ghimj[1012]*K[115]- Ghimj[1013]*K[116]- Ghimj[1014]*K[117] - Ghimj[1015]*K[118]- Ghimj[1016]*K[119]- Ghimj[1017]*K[120]- Ghimj[1018]*K[121]- Ghimj[1019]*K[122]- Ghimj[1020]*K[123] - Ghimj[1021]*K[124]- Ghimj[1022]*K[125];
        K[127] = K[127]- Ghimj[1036]*K[1]- Ghimj[1037]*K[39]- Ghimj[1038]*K[41]- Ghimj[1039]*K[42]- Ghimj[1040]*K[43]- Ghimj[1041]*K[50]  - Ghimj[1042]*K[52]- Ghimj[1043]*K[54]- Ghimj[1044]*K[55]- Ghimj[1045]*K[57]- Ghimj[1046]*K[75]- Ghimj[1047]*K[80]- Ghimj[1048] *K[83]- Ghimj[1049]*K[88]- Ghimj[1050]*K[90]- Ghimj[1051]*K[97]- Ghimj[1052]*K[98]- Ghimj[1053]*K[100]- Ghimj[1054]*K[103] - Ghimj[1055]*K[104]- Ghimj[1056]*K[105]- Ghimj[1057]*K[106]- Ghimj[1058]*K[107]- Ghimj[1059]*K[112]- Ghimj[1060]*K[114] - Ghimj[1061]*K[116]- Ghimj[1062]*K[118]- Ghimj[1063]*K[119]- Ghimj[1064]*K[120]- Ghimj[1065]*K[121]- Ghimj[1066]*K[122] - Ghimj[1067]*K[123]- Ghimj[1068]*K[124]- Ghimj[1069]*K[125]- Ghimj[1070]*K[126];
        K[128] = K[128]- Ghimj[1083]*K[40]- Ghimj[1084]*K[44]- Ghimj[1085]*K[45]- Ghimj[1086]*K[47]- Ghimj[1087]*K[48]- Ghimj[1088]*K[49]  - Ghimj[1089]*K[52]- Ghimj[1090]*K[53]- Ghimj[1091]*K[54]- Ghimj[1092]*K[55]- Ghimj[1093]*K[57]- Ghimj[1094]*K[61]- Ghimj[1095] *K[63]- Ghimj[1096]*K[67]- Ghimj[1097]*K[70]- Ghimj[1098]*K[73]- Ghimj[1099]*K[74]- Ghimj[1100]*K[75]- Ghimj[1101]*K[76] - Ghimj[1102]*K[77]- Ghimj[1103]*K[78]- Ghimj[1104]*K[79]- Ghimj[1105]*K[83]- Ghimj[1106]*K[84]- Ghimj[1107]*K[86]- Ghimj[1108] *K[87]- Ghimj[1109]*K[88]- Ghimj[1110]*K[92]- Ghimj[1111]*K[93]- Ghimj[1112]*K[97]- Ghimj[1113]*K[98]- Ghimj[1114]*K[101] - Ghimj[1115]*K[102]- Ghimj[1116]*K[103]- Ghimj[1117]*K[104]- Ghimj[1118]*K[105]- Ghimj[1119]*K[106]- Ghimj[1120]*K[107] - Ghimj[1121]*K[110]- Ghimj[1122]*K[111]- Ghimj[1123]*K[112]- Ghimj[1124]*K[114]- Ghimj[1125]*K[115]- Ghimj[1126]*K[116] - Ghimj[1127]*K[117]- Ghimj[1128]*K[118]- Ghimj[1129]*K[119]- Ghimj[1130]*K[120]- Ghimj[1131]*K[121]- Ghimj[1132]*K[122] - Ghimj[1133]*K[123]- Ghimj[1134]*K[124]- Ghimj[1135]*K[125]- Ghimj[1136]*K[126]- Ghimj[1137]*K[127];
        K[129] = K[129]- Ghimj[1149]*K[0]- Ghimj[1150]*K[1]- Ghimj[1151]*K[2]- Ghimj[1152]*K[44]- Ghimj[1153]*K[45]- Ghimj[1154]*K[52]- Ghimj[1155]  *K[53]- Ghimj[1156]*K[54]- Ghimj[1157]*K[55]- Ghimj[1158]*K[80]- Ghimj[1159]*K[90]- Ghimj[1160]*K[100]- Ghimj[1161]*K[103] - Ghimj[1162]*K[104]- Ghimj[1163]*K[105]- Ghimj[1164]*K[112]- Ghimj[1165]*K[114]- Ghimj[1166]*K[116]- Ghimj[1167]*K[118] - Ghimj[1168]*K[119]- Ghimj[1169]*K[121]- Ghimj[1170]*K[123]- Ghimj[1171]*K[124]- Ghimj[1172]*K[125]- Ghimj[1173]*K[126] - Ghimj[1174]*K[127]- Ghimj[1175]*K[128];
        K[130] = K[130]- Ghimj[1186]*K[58]- Ghimj[1187]*K[65]- Ghimj[1188]*K[66]- Ghimj[1189]*K[72]- Ghimj[1190]*K[77]- Ghimj[1191]*K[82]  - Ghimj[1192]*K[89]- Ghimj[1193]*K[91]- Ghimj[1194]*K[93]- Ghimj[1195]*K[94]- Ghimj[1196]*K[98]- Ghimj[1197]*K[102]- Ghimj[1198] *K[103]- Ghimj[1199]*K[104]- Ghimj[1200]*K[106]- Ghimj[1201]*K[107]- Ghimj[1202]*K[108]- Ghimj[1203]*K[109]- Ghimj[1204]*K[110] - Ghimj[1205]*K[113]- Ghimj[1206]*K[114]- Ghimj[1207]*K[115]- Ghimj[1208]*K[117]- Ghimj[1209]*K[120]- Ghimj[1210]*K[121] - Ghimj[1211]*K[122]- Ghimj[1212]*K[124]- Ghimj[1213]*K[125]- Ghimj[1214]*K[126]- Ghimj[1215]*K[127]- Ghimj[1216]*K[128] - Ghimj[1217]*K[129];
        K[131] = K[131]- Ghimj[1227]*K[51]- Ghimj[1228]*K[59]- Ghimj[1229]*K[75]- Ghimj[1230]*K[116]- Ghimj[1231]*K[118]- Ghimj[1232]*K[120]  - Ghimj[1233]*K[122]- Ghimj[1234]*K[123]- Ghimj[1235]*K[124]- Ghimj[1236]*K[125]- Ghimj[1237]*K[126]- Ghimj[1238]*K[127] - Ghimj[1239]*K[128]- Ghimj[1240]*K[129]- Ghimj[1241]*K[130];
        K[132] = K[132]- Ghimj[1250]*K[105]- Ghimj[1251]*K[114]- Ghimj[1252]*K[118]- Ghimj[1253]*K[123]- Ghimj[1254]*K[124]- Ghimj[1255]*K[125]  - Ghimj[1256]*K[126]- Ghimj[1257]*K[127]- Ghimj[1258]*K[128]- Ghimj[1259]*K[129]- Ghimj[1260]*K[130]- Ghimj[1261]*K[131];
        K[133] = K[133]- Ghimj[1269]*K[59]- Ghimj[1270]*K[60]- Ghimj[1271]*K[70]- Ghimj[1272]*K[76]- Ghimj[1273]*K[84]- Ghimj[1274]*K[87]  - Ghimj[1275]*K[92]- Ghimj[1276]*K[93]- Ghimj[1277]*K[94]- Ghimj[1278]*K[99]- Ghimj[1279]*K[102]- Ghimj[1280]*K[109]- Ghimj[1281] *K[111]- Ghimj[1282]*K[113]- Ghimj[1283]*K[115]- Ghimj[1284]*K[117]- Ghimj[1285]*K[120]- Ghimj[1286]*K[121]- Ghimj[1287]*K[122] - Ghimj[1288]*K[124]- Ghimj[1289]*K[125]- Ghimj[1290]*K[126]- Ghimj[1291]*K[127]- Ghimj[1292]*K[128]- Ghimj[1293]*K[129] - Ghimj[1294]*K[130]- Ghimj[1295]*K[131]- Ghimj[1296]*K[132];
        K[134] = K[134]- Ghimj[1303]*K[39]- Ghimj[1304]*K[41]- Ghimj[1305]*K[42]- Ghimj[1306]*K[43]- Ghimj[1307]*K[51]- Ghimj[1308]*K[75]  - Ghimj[1309]*K[112]- Ghimj[1310]*K[116]- Ghimj[1311]*K[120]- Ghimj[1312]*K[122]- Ghimj[1313]*K[123]- Ghimj[1314]*K[124] - Ghimj[1315]*K[125]- Ghimj[1316]*K[126]- Ghimj[1317]*K[127]- Ghimj[1318]*K[128]- Ghimj[1319]*K[129]- Ghimj[1320]*K[130] - Ghimj[1321]*K[131]- Ghimj[1322]*K[132]- Ghimj[1323]*K[133];
        K[135] = K[135]- Ghimj[1329]*K[0]- Ghimj[1330]*K[50]- Ghimj[1331]*K[58]- Ghimj[1332]*K[59]- Ghimj[1333]*K[62]- Ghimj[1334]*K[64]  - Ghimj[1335]*K[73]- Ghimj[1336]*K[76]- Ghimj[1337]*K[77]- Ghimj[1338]*K[83]- Ghimj[1339]*K[87]- Ghimj[1340]*K[91]- Ghimj[1341] *K[92]- Ghimj[1342]*K[93]- Ghimj[1343]*K[94]- Ghimj[1344]*K[99]- Ghimj[1345]*K[101]- Ghimj[1346]*K[102]- Ghimj[1347]*K[105] - Ghimj[1348]*K[106]- Ghimj[1349]*K[109]- Ghimj[1350]*K[111]- Ghimj[1351]*K[113]- Ghimj[1352]*K[114]- Ghimj[1353]*K[115] - Ghimj[1354]*K[116]- Ghimj[1355]*K[117]- Ghimj[1356]*K[119]- Ghimj[1357]*K[121]- Ghimj[1358]*K[123]- Ghimj[1359]*K[124] - Ghimj[1360]*K[125]- Ghimj[1361]*K[126]- Ghimj[1362]*K[127]- Ghimj[1363]*K[128]- Ghimj[1364]*K[129]- Ghimj[1365]*K[130] - Ghimj[1366]*K[131]- Ghimj[1367]*K[132]- Ghimj[1368]*K[133]- Ghimj[1369]*K[134];
        K[136] = K[136]- Ghimj[1374]*K[73]- Ghimj[1375]*K[83]- Ghimj[1376]*K[101]- Ghimj[1377]*K[105]- Ghimj[1378]*K[106]- Ghimj[1379]*K[107]  - Ghimj[1380]*K[114]- Ghimj[1381]*K[116]- Ghimj[1382]*K[117]- Ghimj[1383]*K[119]- Ghimj[1384]*K[121]- Ghimj[1385]*K[123] - Ghimj[1386]*K[124]- Ghimj[1387]*K[125]- Ghimj[1388]*K[126]- Ghimj[1389]*K[127]- Ghimj[1390]*K[128]- Ghimj[1391]*K[129] - Ghimj[1392]*K[130]- Ghimj[1393]*K[131]- Ghimj[1394]*K[132]- Ghimj[1395]*K[133]- Ghimj[1396]*K[134]- Ghimj[1397]*K[135];
        K[137] = K[137]- Ghimj[1401]*K[46]- Ghimj[1402]*K[56]- Ghimj[1403]*K[62]- Ghimj[1404]*K[65]- Ghimj[1405]*K[66]- Ghimj[1406]*K[69]  - Ghimj[1407]*K[71]- Ghimj[1408]*K[73]- Ghimj[1409]*K[78]- Ghimj[1410]*K[79]- Ghimj[1411]*K[81]- Ghimj[1412]*K[82]- Ghimj[1413] *K[87]- Ghimj[1414]*K[88]- Ghimj[1415]*K[89]- Ghimj[1416]*K[91]- Ghimj[1417]*K[92]- Ghimj[1418]*K[93]- Ghimj[1419]*K[94] - Ghimj[1420]*K[96]- Ghimj[1421]*K[99]- Ghimj[1422]*K[102]- Ghimj[1423]*K[103]- Ghimj[1424]*K[104]- Ghimj[1425]*K[106] - Ghimj[1426]*K[107]- Ghimj[1427]*K[108]- Ghimj[1428]*K[109]- Ghimj[1429]*K[110]- Ghimj[1430]*K[111]- Ghimj[1431]*K[113] - Ghimj[1432]*K[114]- Ghimj[1433]*K[115]- Ghimj[1434]*K[117]- Ghimj[1435]*K[119]- Ghimj[1436]*K[121]- Ghimj[1437]*K[122] - Ghimj[1438]*K[124]- Ghimj[1439]*K[125]- Ghimj[1440]*K[126]- Ghimj[1441]*K[127]- Ghimj[1442]*K[128]- Ghimj[1443]*K[129] - Ghimj[1444]*K[130]- Ghimj[1445]*K[131]- Ghimj[1446]*K[132]- Ghimj[1447]*K[133]- Ghimj[1448]*K[134]- Ghimj[1449]*K[135] - Ghimj[1450]*K[136];
        K[138] = K[138]- Ghimj[1453]*K[83]- Ghimj[1454]*K[88]- Ghimj[1455]*K[97]- Ghimj[1456]*K[98]- Ghimj[1457]*K[103]- Ghimj[1458]*K[104]  - Ghimj[1459]*K[105]- Ghimj[1460]*K[106]- Ghimj[1461]*K[107]- Ghimj[1462]*K[112]- Ghimj[1463]*K[114]- Ghimj[1464]*K[116] - Ghimj[1465]*K[118]- Ghimj[1466]*K[119]- Ghimj[1467]*K[120]- Ghimj[1468]*K[121]- Ghimj[1469]*K[122]- Ghimj[1470]*K[123] - Ghimj[1471]*K[124]- Ghimj[1472]*K[125]- Ghimj[1473]*K[126]- Ghimj[1474]*K[127]- Ghimj[1475]*K[128]- Ghimj[1476]*K[129] - Ghimj[1477]*K[130]- Ghimj[1478]*K[131]- Ghimj[1479]*K[132]- Ghimj[1480]*K[133]- Ghimj[1481]*K[134]- Ghimj[1482]*K[135] - Ghimj[1483]*K[136]- Ghimj[1484]*K[137];
        K[138] = K[138]/ Ghimj[1485];
        K[137] = (K[137]- Ghimj[1452]*K[138])/(Ghimj[1451]);
        K[136] = (K[136]- Ghimj[1399]*K[137]- Ghimj[1400]*K[138])/(Ghimj[1398]);
        K[135] = (K[135]- Ghimj[1371]*K[136]- Ghimj[1372]*K[137]- Ghimj[1373]*K[138])/(Ghimj[1370]);
        K[134] = (K[134]- Ghimj[1325]*K[135]- Ghimj[1326]*K[136]- Ghimj[1327]*K[137]- Ghimj[1328]*K[138])/(Ghimj[1324]);
        K[133] = (K[133]- Ghimj[1298]*K[134]- Ghimj[1299]*K[135]- Ghimj[1300]*K[136]- Ghimj[1301]*K[137]- Ghimj[1302]*K[138])/(Ghimj[1297]);
        K[132] = (K[132]- Ghimj[1263]*K[133]- Ghimj[1264]*K[134]- Ghimj[1265]*K[135]- Ghimj[1266]*K[136]- Ghimj[1267]*K[137]- Ghimj[1268]  *K[138])/(Ghimj[1262]);
        K[131] = (K[131]- Ghimj[1243]*K[132]- Ghimj[1244]*K[133]- Ghimj[1245]*K[134]- Ghimj[1246]*K[135]- Ghimj[1247]*K[136]- Ghimj[1248]*K[137]  - Ghimj[1249]*K[138])/(Ghimj[1242]);
        K[130] = (K[130]- Ghimj[1219]*K[131]- Ghimj[1220]*K[132]- Ghimj[1221]*K[133]- Ghimj[1222]*K[134]- Ghimj[1223]*K[135]- Ghimj[1224]*K[136]  - Ghimj[1225]*K[137]- Ghimj[1226]*K[138])/(Ghimj[1218]);
        K[129] = (K[129]- Ghimj[1177]*K[130]- Ghimj[1178]*K[131]- Ghimj[1179]*K[132]- Ghimj[1180]*K[133]- Ghimj[1181]*K[134]- Ghimj[1182]*K[135]  - Ghimj[1183]*K[136]- Ghimj[1184]*K[137]- Ghimj[1185]*K[138])/(Ghimj[1176]);
        K[128] = (K[128]- Ghimj[1139]*K[129]- Ghimj[1140]*K[130]- Ghimj[1141]*K[131]- Ghimj[1142]*K[132]- Ghimj[1143]*K[133]- Ghimj[1144]*K[134]  - Ghimj[1145]*K[135]- Ghimj[1146]*K[136]- Ghimj[1147]*K[137]- Ghimj[1148]*K[138])/(Ghimj[1138]);
        K[127] = (K[127]- Ghimj[1072]*K[128]- Ghimj[1073]*K[129]- Ghimj[1074]*K[130]- Ghimj[1075]*K[131]- Ghimj[1076]*K[132]- Ghimj[1077]*K[133]  - Ghimj[1078]*K[134]- Ghimj[1079]*K[135]- Ghimj[1080]*K[136]- Ghimj[1081]*K[137]- Ghimj[1082]*K[138])/(Ghimj[1071]);
        K[126] = (K[126]- Ghimj[1024]*K[127]- Ghimj[1025]*K[128]- Ghimj[1026]*K[129]- Ghimj[1027]*K[130]- Ghimj[1028]*K[131]- Ghimj[1029]*K[132]  - Ghimj[1030]*K[133]- Ghimj[1031]*K[134]- Ghimj[1032]*K[135]- Ghimj[1033]*K[136]- Ghimj[1034]*K[137]- Ghimj[1035]*K[138]) /(Ghimj[1023]);
        K[125] = (K[125]- Ghimj[935]*K[126]- Ghimj[936]*K[127]- Ghimj[937]*K[128]- Ghimj[938]*K[129]- Ghimj[939]*K[130]- Ghimj[940]*K[131]  - Ghimj[941]*K[132]- Ghimj[942]*K[133]- Ghimj[943]*K[134]- Ghimj[944]*K[135]- Ghimj[945]*K[136]- Ghimj[946]*K[137]- Ghimj[947] *K[138])/(Ghimj[934]);
        K[124] = (K[124]- Ghimj[897]*K[125]- Ghimj[898]*K[126]- Ghimj[899]*K[127]- Ghimj[900]*K[128]- Ghimj[901]*K[129]- Ghimj[902]*K[130]  - Ghimj[903]*K[131]- Ghimj[904]*K[132]- Ghimj[905]*K[133]- Ghimj[906]*K[135]- Ghimj[907]*K[136]- Ghimj[908]*K[137]- Ghimj[909] *K[138])/(Ghimj[896]);
        K[123] = (K[123]- Ghimj[870]*K[124]- Ghimj[871]*K[125]- Ghimj[872]*K[126]- Ghimj[873]*K[127]- Ghimj[874]*K[128]- Ghimj[875]*K[129]  - Ghimj[876]*K[130]- Ghimj[877]*K[131]- Ghimj[878]*K[132]- Ghimj[879]*K[133]- Ghimj[880]*K[134]- Ghimj[881]*K[135]- Ghimj[882] *K[136]- Ghimj[883]*K[137]- Ghimj[884]*K[138])/(Ghimj[869]);
        K[122] = (K[122]- Ghimj[848]*K[124]- Ghimj[849]*K[125]- Ghimj[850]*K[126]- Ghimj[851]*K[127]- Ghimj[852]*K[128]- Ghimj[853]*K[129]  - Ghimj[854]*K[130]- Ghimj[855]*K[131]- Ghimj[856]*K[133]- Ghimj[857]*K[135]- Ghimj[858]*K[136]- Ghimj[859]*K[137]- Ghimj[860] *K[138])/(Ghimj[847]);
        K[121] = (K[121]- Ghimj[822]*K[124]- Ghimj[823]*K[125]- Ghimj[824]*K[126]- Ghimj[825]*K[127]- Ghimj[826]*K[129]- Ghimj[827]*K[133]  - Ghimj[828]*K[135]- Ghimj[829]*K[136]- Ghimj[830]*K[137])/(Ghimj[821]);
        K[120] = (K[120]- Ghimj[788]*K[122]- Ghimj[789]*K[124]- Ghimj[790]*K[126]- Ghimj[791]*K[127]- Ghimj[792]*K[128]- Ghimj[793]*K[130]  - Ghimj[794]*K[133]- Ghimj[795]*K[135]- Ghimj[796]*K[136]- Ghimj[797]*K[137])/(Ghimj[787]);
        K[119] = (K[119]- Ghimj[768]*K[121]- Ghimj[769]*K[124]- Ghimj[770]*K[125]- Ghimj[771]*K[126]- Ghimj[772]*K[127]- Ghimj[773]*K[129]  - Ghimj[774]*K[133]- Ghimj[775]*K[136]- Ghimj[776]*K[137])/(Ghimj[767]);
        K[118] = (K[118]- Ghimj[746]*K[123]- Ghimj[747]*K[125]- Ghimj[748]*K[126]- Ghimj[749]*K[127]- Ghimj[750]*K[128]- Ghimj[751]*K[129]  - Ghimj[752]*K[131]- Ghimj[753]*K[132]- Ghimj[754]*K[134]- Ghimj[755]*K[135]- Ghimj[756]*K[137]- Ghimj[757]*K[138])/(Ghimj[745]);
        K[117] = (K[117]- Ghimj[732]*K[121]- Ghimj[733]*K[124]- Ghimj[734]*K[125]- Ghimj[735]*K[126]- Ghimj[736]*K[127]- Ghimj[737]*K[129]  - Ghimj[738]*K[133]- Ghimj[739]*K[136]- Ghimj[740]*K[137])/(Ghimj[731]);
        K[116] = (K[116]- Ghimj[715]*K[123]- Ghimj[716]*K[127]- Ghimj[717]*K[128]- Ghimj[718]*K[131]- Ghimj[719]*K[134]- Ghimj[720]*K[135]  - Ghimj[721]*K[138])/(Ghimj[714]);
        K[115] = (K[115]- Ghimj[707]*K[124]- Ghimj[708]*K[126]- Ghimj[709]*K[127]- Ghimj[710]*K[129]- Ghimj[711]*K[133]- Ghimj[712]*K[136]  - Ghimj[713]*K[137])/(Ghimj[706]);
        K[114] = (K[114]- Ghimj[698]*K[126]- Ghimj[699]*K[127]- Ghimj[700]*K[129]- Ghimj[701]*K[132]- Ghimj[702]*K[136])/(Ghimj[697]);
        K[113] = (K[113]- Ghimj[690]*K[124]- Ghimj[691]*K[125]- Ghimj[692]*K[126]- Ghimj[693]*K[133]- Ghimj[694]*K[135]- Ghimj[695]*K[136]  - Ghimj[696]*K[137])/(Ghimj[689]);
        K[112] = (K[112]- Ghimj[678]*K[116]- Ghimj[679]*K[123]- Ghimj[680]*K[126]- Ghimj[681]*K[128]- Ghimj[682]*K[134]- Ghimj[683]*K[137]  - Ghimj[684]*K[138])/(Ghimj[677]);
        K[111] = (K[111]- Ghimj[670]*K[115]- Ghimj[671]*K[124]- Ghimj[672]*K[125]- Ghimj[673]*K[126]- Ghimj[674]*K[133]- Ghimj[675]*K[136]  - Ghimj[676]*K[137])/(Ghimj[669]);
        K[110] = (K[110]- Ghimj[660]*K[124]- Ghimj[661]*K[125]- Ghimj[662]*K[126]- Ghimj[663]*K[133]- Ghimj[664]*K[136]- Ghimj[665]*K[137])  /(Ghimj[659]);
        K[109] = (K[109]- Ghimj[649]*K[124]- Ghimj[650]*K[125]- Ghimj[651]*K[126]- Ghimj[652]*K[133]- Ghimj[653]*K[136]- Ghimj[654]*K[137])  /(Ghimj[648]);
        K[108] = (K[108]- Ghimj[637]*K[109]- Ghimj[638]*K[113]- Ghimj[639]*K[115]- Ghimj[640]*K[124]- Ghimj[641]*K[125]- Ghimj[642]*K[126]  - Ghimj[643]*K[133]- Ghimj[644]*K[135]- Ghimj[645]*K[136]- Ghimj[646]*K[137])/(Ghimj[636]);
        K[107] = (K[107]- Ghimj[627]*K[124]- Ghimj[628]*K[126]- Ghimj[629]*K[136])/(Ghimj[626]);
        K[106] = (K[106]- Ghimj[623]*K[124]- Ghimj[624]*K[126]- Ghimj[625]*K[136])/(Ghimj[622]);
        K[105] = (K[105]- Ghimj[617]*K[128]- Ghimj[618]*K[129]- Ghimj[619]*K[132]- Ghimj[620]*K[135]- Ghimj[621]*K[138])/(Ghimj[616]);
        K[104] = (K[104]- Ghimj[611]*K[125]- Ghimj[612]*K[126]- Ghimj[613]*K[127]- Ghimj[614]*K[129]- Ghimj[615]*K[137])/(Ghimj[610]);
        K[103] = (K[103]- Ghimj[606]*K[124]- Ghimj[607]*K[126]- Ghimj[608]*K[127]- Ghimj[609]*K[129])/(Ghimj[605]);
        K[102] = (K[102]- Ghimj[601]*K[125]- Ghimj[602]*K[126]- Ghimj[603]*K[133]- Ghimj[604]*K[137])/(Ghimj[600]);
        K[101] = (K[101]- Ghimj[587]*K[105]- Ghimj[588]*K[114]- Ghimj[589]*K[116]- Ghimj[590]*K[119]- Ghimj[591]*K[123]- Ghimj[592]*K[126]  - Ghimj[593]*K[128]- Ghimj[594]*K[130]- Ghimj[595]*K[135]- Ghimj[596]*K[136]- Ghimj[597]*K[138])/(Ghimj[586]);
        K[100] = (K[100]- Ghimj[574]*K[105]- Ghimj[575]*K[112]- Ghimj[576]*K[116]- Ghimj[577]*K[118]- Ghimj[578]*K[123]- Ghimj[579]*K[126]  - Ghimj[580]*K[127]- Ghimj[581]*K[129]- Ghimj[582]*K[132]- Ghimj[583]*K[134]- Ghimj[584]*K[138])/(Ghimj[573]);
        K[99] = (K[99]- Ghimj[566]*K[102]- Ghimj[567]*K[111]- Ghimj[568]*K[125]- Ghimj[569]*K[126]- Ghimj[570]*K[133]- Ghimj[571]*K[137])  /(Ghimj[565]);
        K[98] = (K[98]- Ghimj[558]*K[107]- Ghimj[559]*K[120]- Ghimj[560]*K[124]- Ghimj[561]*K[126]- Ghimj[562]*K[127])/(Ghimj[557]);
        K[97] = (K[97]- Ghimj[550]*K[98]- Ghimj[551]*K[120]- Ghimj[552]*K[122]- Ghimj[553]*K[126]- Ghimj[554]*K[127]- Ghimj[555]*K[130]- Ghimj[556]  *K[137])/(Ghimj[549]);
        K[96] = (K[96]- Ghimj[539]*K[107]- Ghimj[540]*K[108]- Ghimj[541]*K[109]- Ghimj[542]*K[110]- Ghimj[543]*K[113]- Ghimj[544]*K[124]  - Ghimj[545]*K[125]- Ghimj[546]*K[126]- Ghimj[547]*K[133]- Ghimj[548]*K[137])/(Ghimj[538]);
        K[95] = (K[95]- Ghimj[515]*K[96]- Ghimj[516]*K[98]- Ghimj[517]*K[103]- Ghimj[518]*K[106]- Ghimj[519]*K[107]- Ghimj[520]*K[109]- Ghimj[521]  *K[110]- Ghimj[522]*K[113]- Ghimj[523]*K[119]- Ghimj[524]*K[121]- Ghimj[525]*K[124]- Ghimj[526]*K[125]- Ghimj[527]*K[126] - Ghimj[528]*K[127]- Ghimj[529]*K[129]- Ghimj[530]*K[130]- Ghimj[531]*K[133]- Ghimj[532]*K[135]- Ghimj[533]*K[136]- Ghimj[534] *K[137])/(Ghimj[514]);
        K[94] = (K[94]- Ghimj[506]*K[125]- Ghimj[507]*K[126]- Ghimj[508]*K[133]- Ghimj[509]*K[137])/(Ghimj[505]);
        K[93] = (K[93]- Ghimj[498]*K[125]- Ghimj[499]*K[126]- Ghimj[500]*K[133]- Ghimj[501]*K[137])/(Ghimj[497]);
        K[92] = (K[92]- Ghimj[490]*K[124]- Ghimj[491]*K[126]- Ghimj[492]*K[133]- Ghimj[493]*K[135]- Ghimj[494]*K[137])/(Ghimj[489]);
        K[91] = (K[91]- Ghimj[482]*K[106]- Ghimj[483]*K[109]- Ghimj[484]*K[126]- Ghimj[485]*K[133]- Ghimj[486]*K[136])/(Ghimj[481]);
        K[90] = (K[90]- Ghimj[470]*K[100]- Ghimj[471]*K[105]- Ghimj[472]*K[112]- Ghimj[473]*K[116]- Ghimj[474]*K[118]- Ghimj[475]*K[123]  - Ghimj[476]*K[127]- Ghimj[477]*K[129]- Ghimj[478]*K[132]- Ghimj[479]*K[134]- Ghimj[480]*K[138])/(Ghimj[469]);
        K[89] = (K[89]- Ghimj[458]*K[93]- Ghimj[459]*K[94]- Ghimj[460]*K[102]- Ghimj[461]*K[107]- Ghimj[462]*K[109]- Ghimj[463]*K[113]- Ghimj[464]  *K[117]- Ghimj[465]*K[124]- Ghimj[466]*K[125]- Ghimj[467]*K[126])/(Ghimj[457]);
        K[88] = (K[88]- Ghimj[451]*K[103]- Ghimj[452]*K[106]- Ghimj[453]*K[124]- Ghimj[454]*K[126]- Ghimj[455]*K[127]- Ghimj[456]*K[137])  /(Ghimj[450]);
        K[87] = (K[87]- Ghimj[445]*K[92]- Ghimj[446]*K[124]- Ghimj[447]*K[126]- Ghimj[448]*K[135]- Ghimj[449]*K[137])/(Ghimj[444]);
        K[86] = (K[86]- Ghimj[437]*K[93]- Ghimj[438]*K[125]- Ghimj[439]*K[126]- Ghimj[440]*K[133]- Ghimj[441]*K[137])/(Ghimj[436]);
        K[85] = (K[85]- Ghimj[428]*K[102]- Ghimj[429]*K[111]- Ghimj[430]*K[125]- Ghimj[431]*K[126]- Ghimj[432]*K[133]- Ghimj[433]*K[137])  /(Ghimj[427]);
        K[84] = (K[84]- Ghimj[422]*K[92]- Ghimj[423]*K[124]- Ghimj[424]*K[135]- Ghimj[425]*K[137])/(Ghimj[421]);
        K[83] = (K[83]- Ghimj[417]*K[128]- Ghimj[418]*K[135]- Ghimj[419]*K[136]- Ghimj[420]*K[138])/(Ghimj[416]);
        K[82] = (K[82]- Ghimj[413]*K[113]- Ghimj[414]*K[126]- Ghimj[415]*K[137])/(Ghimj[412]);
        K[81] = (K[81]- Ghimj[406]*K[114]- Ghimj[407]*K[124]- Ghimj[408]*K[126]- Ghimj[409]*K[127]- Ghimj[410]*K[129]- Ghimj[411]*K[136])  /(Ghimj[405]);
        K[80] = (K[80]- Ghimj[398]*K[90]- Ghimj[399]*K[112]- Ghimj[400]*K[116]- Ghimj[401]*K[127]- Ghimj[402]*K[129]- Ghimj[403]*K[134]- Ghimj[404]  *K[138])/(Ghimj[397]);
        K[79] = (K[79]- Ghimj[394]*K[102]- Ghimj[395]*K[126]- Ghimj[396]*K[137])/(Ghimj[393]);
        K[78] = (K[78]- Ghimj[387]*K[103]- Ghimj[388]*K[106]- Ghimj[389]*K[107]- Ghimj[390]*K[110]- Ghimj[391]*K[124]- Ghimj[392]*K[126])  /(Ghimj[386]);
        K[77] = (K[77]- Ghimj[383]*K[121]- Ghimj[384]*K[126]- Ghimj[385]*K[135])/(Ghimj[382]);
        K[76] = (K[76]- Ghimj[378]*K[87]- Ghimj[379]*K[126]- Ghimj[380]*K[133]- Ghimj[381]*K[135])/(Ghimj[377]);
        K[75] = (K[75]- Ghimj[375]*K[120]- Ghimj[376]*K[126])/(Ghimj[374]);
        K[74] = (K[74]- Ghimj[369]*K[117]- Ghimj[370]*K[121]- Ghimj[371]*K[125]- Ghimj[372]*K[126]- Ghimj[373]*K[137])/(Ghimj[368]);
        K[73] = (K[73]- Ghimj[365]*K[126]- Ghimj[366]*K[135]- Ghimj[367]*K[137])/(Ghimj[364]);
        K[72] = (K[72]- Ghimj[361]*K[94]- Ghimj[362]*K[126]- Ghimj[363]*K[137])/(Ghimj[360]);
        K[71] = (K[71]- Ghimj[357]*K[117]- Ghimj[358]*K[126]- Ghimj[359]*K[137])/(Ghimj[356]);
        K[70] = (K[70]- Ghimj[353]*K[84]- Ghimj[354]*K[87]- Ghimj[355]*K[126])/(Ghimj[352]);
        K[69] = (K[69]- Ghimj[348]*K[93]- Ghimj[349]*K[126]- Ghimj[350]*K[137])/(Ghimj[347]);
        K[68] = (K[68]- Ghimj[344]*K[99]- Ghimj[345]*K[126]- Ghimj[346]*K[137])/(Ghimj[343]);
        K[67] = (K[67]- Ghimj[340]*K[115]- Ghimj[341]*K[126]- Ghimj[342]*K[137])/(Ghimj[339]);
        K[66] = (K[66]- Ghimj[336]*K[109]- Ghimj[337]*K[126]- Ghimj[338]*K[137])/(Ghimj[335]);
        K[65] = (K[65]- Ghimj[332]*K[114]- Ghimj[333]*K[126]- Ghimj[334]*K[132])/(Ghimj[331]);
        K[64] = (K[64]- Ghimj[328]*K[113]- Ghimj[329]*K[126]- Ghimj[330]*K[135])/(Ghimj[327]);
        K[63] = (K[63]- Ghimj[324]*K[121]- Ghimj[325]*K[126]- Ghimj[326]*K[137])/(Ghimj[323]);
        K[62] = (K[62]- Ghimj[320]*K[93]- Ghimj[321]*K[126]- Ghimj[322]*K[133])/(Ghimj[319]);
        K[61] = (K[61]- Ghimj[316]*K[70]- Ghimj[317]*K[87]- Ghimj[318]*K[126])/(Ghimj[315]);
        K[60] = (K[60]- Ghimj[311]*K[92]- Ghimj[312]*K[120]- Ghimj[313]*K[133]- Ghimj[314]*K[135])/(Ghimj[310]);
        K[59] = (K[59]- Ghimj[307]*K[133]- Ghimj[308]*K[135])/(Ghimj[306]);
        K[58] = (K[58]- Ghimj[304]*K[91]- Ghimj[305]*K[126])/(Ghimj[303]);
        K[57] = (K[57]- Ghimj[301]*K[120]- Ghimj[302]*K[126])/(Ghimj[300]);
        K[56] = (K[56]- Ghimj[297]*K[65]- Ghimj[298]*K[81]- Ghimj[299]*K[126])/(Ghimj[296]);
        K[55] = (K[55]- Ghimj[295]*K[126])/(Ghimj[294]);
        K[54] = (K[54]- Ghimj[293]*K[126])/(Ghimj[292]);
        K[53] = (K[53]- Ghimj[291]*K[126])/(Ghimj[290]);
        K[52] = (K[52]- Ghimj[289]*K[126])/(Ghimj[288]);
        K[51] = (K[51]- Ghimj[286]*K[132]- Ghimj[287]*K[134])/(Ghimj[285]);
        K[50] = (K[50]- Ghimj[283]*K[83]- Ghimj[284]*K[138])/(Ghimj[282]);
        K[49] = (K[49]- Ghimj[281]*K[126])/(Ghimj[280]);
        K[48] = (K[48]- Ghimj[279]*K[126])/(Ghimj[278]);
        K[47] = (K[47]- Ghimj[277]*K[126])/(Ghimj[276]);
        K[46] = (K[46]- Ghimj[273]*K[81]- Ghimj[274]*K[124]- Ghimj[275]*K[137])/(Ghimj[272]);
        K[45] = (K[45]- Ghimj[271]*K[126])/(Ghimj[270]);
        K[44] = (K[44]- Ghimj[269]*K[126])/(Ghimj[268]);
        K[43] = (K[43]- Ghimj[267]*K[120])/(Ghimj[266]);
        K[42] = (K[42]- Ghimj[265]*K[120])/(Ghimj[264]);
        K[41] = (K[41]- Ghimj[263]*K[120])/(Ghimj[262]);
        K[40] = (K[40]- Ghimj[261]*K[126])/(Ghimj[260]);
        K[39] = (K[39]- Ghimj[259]*K[134])/(Ghimj[258]);
        K[38] = (K[38]- Ghimj[256]*K[68]- Ghimj[257]*K[126])/(Ghimj[255]);
        K[37] = (K[37]- Ghimj[252]*K[52]- Ghimj[253]*K[54]- Ghimj[254]*K[55])/(Ghimj[251]);
        K[36] = (K[36]- Ghimj[245]*K[44]- Ghimj[246]*K[45]- Ghimj[247]*K[52]- Ghimj[248]*K[54]- Ghimj[249]*K[55]- Ghimj[250]*K[126])/(Ghimj[244]);
        K[35] = (K[35]- Ghimj[234]*K[93]- Ghimj[235]*K[94]- Ghimj[236]*K[99]- Ghimj[237]*K[102]- Ghimj[238]*K[109]- Ghimj[239]*K[113]- Ghimj[240]  *K[115]- Ghimj[241]*K[117]- Ghimj[242]*K[121]- Ghimj[243]*K[133])/(Ghimj[233]);
        K[34] = (K[34]- Ghimj[207]*K[50]- Ghimj[208]*K[51]- Ghimj[209]*K[59]- Ghimj[210]*K[60]- Ghimj[211]*K[65]- Ghimj[212]*K[73]- Ghimj[213]  *K[76]- Ghimj[214]*K[93]- Ghimj[215]*K[94]- Ghimj[216]*K[99]- Ghimj[217]*K[100]- Ghimj[218]*K[101]- Ghimj[219]*K[102]- Ghimj[220] *K[109]- Ghimj[221]*K[113]- Ghimj[222]*K[114]- Ghimj[223]*K[115]- Ghimj[224]*K[117]- Ghimj[225]*K[121]- Ghimj[226]*K[122] - Ghimj[227]*K[125]- Ghimj[228]*K[126]- Ghimj[229]*K[127]- Ghimj[230]*K[129]- Ghimj[231]*K[133]- Ghimj[232]*K[137])/(Ghimj[206]);
        K[33] = (K[33]- Ghimj[203]*K[125]- Ghimj[204]*K[133])/(Ghimj[202]);
        K[32] = (K[32]- Ghimj[195]*K[41]- Ghimj[196]*K[42]- Ghimj[197]*K[43]- Ghimj[198]*K[57]- Ghimj[199]*K[75]- Ghimj[200]*K[120]- Ghimj[201]  *K[126])/(Ghimj[194]);
        K[31] = (K[31]- Ghimj[191]*K[53]- Ghimj[192]*K[126])/(Ghimj[190]);
        K[30] = (K[30]- Ghimj[186]*K[133]- Ghimj[187]*K[137])/(Ghimj[185]);
        K[29] = (K[29]- Ghimj[183]*K[124]- Ghimj[184]*K[126])/(Ghimj[182]);
        K[28] = (K[28]- Ghimj[171]*K[103]- Ghimj[172]*K[106]- Ghimj[173]*K[107]- Ghimj[174]*K[110]- Ghimj[175]*K[117]- Ghimj[176]*K[119]  - Ghimj[177]*K[121]- Ghimj[178]*K[124]- Ghimj[179]*K[125]- Ghimj[180]*K[130]- Ghimj[181]*K[136])/(Ghimj[170]);
        K[27] = (K[27]- Ghimj[164]*K[60]- Ghimj[165]*K[98]- Ghimj[166]*K[120]- Ghimj[167]*K[124]- Ghimj[168]*K[128]- Ghimj[169]*K[131])  /(Ghimj[163]);
        K[26] = (K[26]- Ghimj[149]*K[83]- Ghimj[150]*K[84]- Ghimj[151]*K[87]- Ghimj[152]*K[92]- Ghimj[153]*K[105]- Ghimj[154]*K[116]- Ghimj[155]  *K[123]- Ghimj[156]*K[124]- Ghimj[157]*K[128]- Ghimj[158]*K[131]- Ghimj[159]*K[135]- Ghimj[160]*K[136]- Ghimj[161]*K[137] - Ghimj[162]*K[138])/(Ghimj[148]);
        K[25] = (K[25]- Ghimj[141]*K[97]- Ghimj[142]*K[120]- Ghimj[143]*K[122]- Ghimj[144]*K[124]- Ghimj[145]*K[126]- Ghimj[146]*K[131]- Ghimj[147]  *K[137])/(Ghimj[140]);
        K[24] = (K[24]- Ghimj[124]*K[39]- Ghimj[125]*K[57]- Ghimj[126]*K[75]- Ghimj[127]*K[83]- Ghimj[128]*K[105]- Ghimj[129]*K[112]- Ghimj[130]  *K[116]- Ghimj[131]*K[118]- Ghimj[132]*K[120]- Ghimj[133]*K[123]- Ghimj[134]*K[125]- Ghimj[135]*K[126]- Ghimj[136]*K[131] - Ghimj[137]*K[132]- Ghimj[138]*K[134]- Ghimj[139]*K[138])/(Ghimj[123]);
        K[23] = (K[23]- Ghimj[113]*K[105]- Ghimj[114]*K[112]- Ghimj[115]*K[116]- Ghimj[116]*K[118]- Ghimj[117]*K[123]- Ghimj[118]*K[125]  - Ghimj[119]*K[131]- Ghimj[120]*K[132]- Ghimj[121]*K[134]- Ghimj[122]*K[138])/(Ghimj[112]);
        K[22] = (K[22]- Ghimj[76]*K[39]- Ghimj[77]*K[57]- Ghimj[78]*K[60]- Ghimj[79]*K[75]- Ghimj[80]*K[83]- Ghimj[81]*K[84]- Ghimj[82]*K[87]  - Ghimj[83]*K[92]- Ghimj[84]*K[97]- Ghimj[85]*K[98]- Ghimj[86]*K[103]- Ghimj[87]*K[105]- Ghimj[88]*K[106]- Ghimj[89]*K[107]- Ghimj[90] *K[110]- Ghimj[91]*K[112]- Ghimj[92]*K[116]- Ghimj[93]*K[117]- Ghimj[94]*K[118]- Ghimj[95]*K[119]- Ghimj[96]*K[120]- Ghimj[97] *K[121]- Ghimj[98]*K[122]- Ghimj[99]*K[123]- Ghimj[100]*K[124]- Ghimj[101]*K[125]- Ghimj[102]*K[126]- Ghimj[103]*K[128]- Ghimj[104] *K[130]- Ghimj[105]*K[131]- Ghimj[106]*K[132]- Ghimj[107]*K[134]- Ghimj[108]*K[135]- Ghimj[109]*K[136]- Ghimj[110]*K[137] - Ghimj[111]*K[138])/(Ghimj[75]);
        K[21] = (K[21]- Ghimj[73]*K[120]- Ghimj[74]*K[128])/(Ghimj[72]);
        K[20] = (K[20]- Ghimj[70]*K[124]- Ghimj[71]*K[137])/(Ghimj[69]);
        K[19] = K[19]/ Ghimj[68];
        K[18] = (K[18]- Ghimj[65]*K[120]- Ghimj[66]*K[126])/(Ghimj[64]);
        K[17] = (K[17]- Ghimj[63]*K[120])/(Ghimj[62]);
        K[16] = (K[16]- Ghimj[61]*K[120])/(Ghimj[60]);
        K[15] = (K[15]- Ghimj[59]*K[120])/(Ghimj[58]);
        K[14] = (K[14]- Ghimj[53]*K[15]- Ghimj[54]*K[16]- Ghimj[55]*K[17]- Ghimj[56]*K[18]- Ghimj[57]*K[120])/(Ghimj[52]);
        K[13] = (K[13]- Ghimj[49]*K[83])/(Ghimj[48]);
        K[12] = (K[12]- Ghimj[47]*K[83])/(Ghimj[46]);
        K[11] = (K[11]- Ghimj[44]*K[56]- Ghimj[45]*K[126])/(Ghimj[43]);
        K[10] = (K[10]- Ghimj[39]*K[46]- Ghimj[40]*K[65]- Ghimj[41]*K[126]- Ghimj[42]*K[137])/(Ghimj[38]);
        K[9] = (K[9]- Ghimj[30]*K[42]- Ghimj[31]*K[43]- Ghimj[32]*K[52]- Ghimj[33]*K[54]- Ghimj[34]*K[55]- Ghimj[35]*K[75]- Ghimj[36]*K[120]  - Ghimj[37]*K[126])/(Ghimj[29]);
        K[8] = (K[8]- Ghimj[26]*K[42]- Ghimj[27]*K[43]- Ghimj[28]*K[120])/(Ghimj[25]);
        K[7] = (K[7]- Ghimj[10]*K[41]- Ghimj[11]*K[42]- Ghimj[12]*K[43]- Ghimj[13]*K[44]- Ghimj[14]*K[45]- Ghimj[15]*K[52]- Ghimj[16]*K[53]- Ghimj[17]  *K[54]- Ghimj[18]*K[55]- Ghimj[19]*K[57]- Ghimj[20]*K[75]- Ghimj[21]*K[120]- Ghimj[22]*K[126])/(Ghimj[9]);
        K[6] = K[6]/ Ghimj[6];
        K[5] = K[5]/ Ghimj[5];
        K[4] = K[4]/ Ghimj[4];
        K[3] = K[3]/ Ghimj[3];
        K[2] = K[2]/ Ghimj[2];
        K[1] = K[1]/ Ghimj[1];
        K[0] = K[0]/ Ghimj[0];
}

__device__ void ros_Solve(double * __restrict__ Ghimj, double * __restrict__ K, int &Nsol, const int istage, const int ros_S)
{

    #pragma unroll 4 
    for (int i=0;i<LU_NONZERO-16;i+=16){
        prefetch_ll1(&Ghimj[i]);
    }

    kppSolve(Ghimj, K, istage, ros_S);
    Nsol++;
}

__device__ void kppDecomp(double *Ghimj, int VL_GLO)
{
    double a=0.0;

 double dummy, W_0, W_1, W_2, W_3, W_4, W_5, W_6, W_7, W_8, W_9, W_10, W_11, W_12, W_13, W_14, W_15, W_16, W_17, W_18, W_19, W_20, W_21, W_22, W_23, W_24, W_25, W_26, W_27, W_28, W_29, W_30, W_31, W_32, W_33, W_34, W_35, W_36, W_37, W_38, W_39, W_40, W_41, W_42, W_43, W_44, W_45, W_46, W_47, W_48, W_49, W_50, W_51, W_52, W_53, W_54, W_55, W_56, W_57, W_58, W_59, W_60, W_61, W_62, W_63, W_64, W_65, W_66, W_67, W_68, W_69, W_70, W_71, W_72, W_73, W_74, W_75, W_76, W_77, W_78, W_79, W_80, W_81, W_82, W_83, W_84, W_85, W_86, W_87, W_88, W_89, W_90, W_91, W_92, W_93, W_94, W_95, W_96, W_97, W_98, W_99, W_100, W_101, W_102, W_103, W_104, W_105, W_106, W_107, W_108, W_109, W_110, W_111, W_112, W_113, W_114, W_115, W_116, W_117, W_118, W_119, W_120, W_121, W_122, W_123, W_124, W_125, W_126, W_127, W_128, W_129, W_130, W_131, W_132, W_133, W_134, W_135, W_136, W_137, W_138, W_139, W_140, W_141;

        W_1 = Ghimj[7];
        W_2 = Ghimj[8];
        W_7 = Ghimj[9];
        W_41 = Ghimj[10];
        W_42 = Ghimj[11];
        W_43 = Ghimj[12];
        W_44 = Ghimj[13];
        W_45 = Ghimj[14];
        W_52 = Ghimj[15];
        W_53 = Ghimj[16];
        W_54 = Ghimj[17];
        W_55 = Ghimj[18];
        W_57 = Ghimj[19];
        W_75 = Ghimj[20];
        W_120 = Ghimj[21];
        W_126 = Ghimj[22];
        a = - W_1/ Ghimj[1];
        W_1 = -a;
        a = - W_2/ Ghimj[2];
        W_2 = -a;
        Ghimj[7] = W_1;
        Ghimj[8] = W_2;
        Ghimj[9] = W_7;
        Ghimj[10] = W_41;
        Ghimj[11] = W_42;
        Ghimj[12] = W_43;
        Ghimj[13] = W_44;
        Ghimj[14] = W_45;
        Ghimj[15] = W_52;
        Ghimj[16] = W_53;
        Ghimj[17] = W_54;
        Ghimj[18] = W_55;
        Ghimj[19] = W_57;
        Ghimj[20] = W_75;
        Ghimj[21] = W_120;
        Ghimj[22] = W_126;
        W_1 = Ghimj[23];
        W_2 = Ghimj[24];
        W_8 = Ghimj[25];
        W_42 = Ghimj[26];
        W_43 = Ghimj[27];
        W_120 = Ghimj[28];
        a = - W_1/ Ghimj[1];
        W_1 = -a;
        a = - W_2/ Ghimj[2];
        W_2 = -a;
        Ghimj[23] = W_1;
        Ghimj[24] = W_2;
        Ghimj[25] = W_8;
        Ghimj[26] = W_42;
        Ghimj[27] = W_43;
        Ghimj[28] = W_120;
        W_5 = Ghimj[50];
        W_6 = Ghimj[51];
        W_14 = Ghimj[52];
        W_15 = Ghimj[53];
        W_16 = Ghimj[54];
        W_17 = Ghimj[55];
        W_18 = Ghimj[56];
        W_120 = Ghimj[57];
        a = - W_5/ Ghimj[5];
        W_5 = -a;
        a = - W_6/ Ghimj[6];
        W_6 = -a;
        Ghimj[50] = W_5;
        Ghimj[51] = W_6;
        Ghimj[52] = W_14;
        Ghimj[53] = W_15;
        Ghimj[54] = W_16;
        Ghimj[55] = W_17;
        Ghimj[56] = W_18;
        Ghimj[57] = W_120;
        W_4 = Ghimj[67];
        W_19 = Ghimj[68];
        a = - W_4/ Ghimj[4];
        W_4 = -a;
        Ghimj[67] = W_4;
        Ghimj[68] = W_19;
        W_1 = Ghimj[188];
        W_2 = Ghimj[189];
        W_31 = Ghimj[190];
        W_53 = Ghimj[191];
        W_126 = Ghimj[192];
        a = - W_1/ Ghimj[1];
        W_1 = -a;
        a = - W_2/ Ghimj[2];
        W_2 = -a;
        Ghimj[188] = W_1;
        Ghimj[189] = W_2;
        Ghimj[190] = W_31;
        Ghimj[191] = W_53;
        Ghimj[192] = W_126;
        W_1 = Ghimj[193];
        W_32 = Ghimj[194];
        W_41 = Ghimj[195];
        W_42 = Ghimj[196];
        W_43 = Ghimj[197];
        W_57 = Ghimj[198];
        W_75 = Ghimj[199];
        W_120 = Ghimj[200];
        W_126 = Ghimj[201];
        a = - W_1/ Ghimj[1];
        W_1 = -a;
        Ghimj[193] = W_1;
        Ghimj[194] = W_32;
        Ghimj[195] = W_41;
        Ghimj[196] = W_42;
        Ghimj[197] = W_43;
        Ghimj[198] = W_57;
        Ghimj[199] = W_75;
        Ghimj[200] = W_120;
        Ghimj[201] = W_126;
        W_0 = Ghimj[205];
        W_34 = Ghimj[206];
        W_50 = Ghimj[207];
        W_51 = Ghimj[208];
        W_59 = Ghimj[209];
        W_60 = Ghimj[210];
        W_65 = Ghimj[211];
        W_73 = Ghimj[212];
        W_76 = Ghimj[213];
        W_93 = Ghimj[214];
        W_94 = Ghimj[215];
        W_99 = Ghimj[216];
        W_100 = Ghimj[217];
        W_101 = Ghimj[218];
        W_102 = Ghimj[219];
        W_109 = Ghimj[220];
        W_113 = Ghimj[221];
        W_114 = Ghimj[222];
        W_115 = Ghimj[223];
        W_117 = Ghimj[224];
        W_121 = Ghimj[225];
        W_122 = Ghimj[226];
        W_125 = Ghimj[227];
        W_126 = Ghimj[228];
        W_127 = Ghimj[229];
        W_129 = Ghimj[230];
        W_133 = Ghimj[231];
        W_137 = Ghimj[232];
        a = - W_0/ Ghimj[0];
        W_0 = -a;
        Ghimj[205] = W_0;
        Ghimj[206] = W_34;
        Ghimj[207] = W_50;
        Ghimj[208] = W_51;
        Ghimj[209] = W_59;
        Ghimj[210] = W_60;
        Ghimj[211] = W_65;
        Ghimj[212] = W_73;
        Ghimj[213] = W_76;
        Ghimj[214] = W_93;
        Ghimj[215] = W_94;
        Ghimj[216] = W_99;
        Ghimj[217] = W_100;
        Ghimj[218] = W_101;
        Ghimj[219] = W_102;
        Ghimj[220] = W_109;
        Ghimj[221] = W_113;
        Ghimj[222] = W_114;
        Ghimj[223] = W_115;
        Ghimj[224] = W_117;
        Ghimj[225] = W_121;
        Ghimj[226] = W_122;
        Ghimj[227] = W_125;
        Ghimj[228] = W_126;
        Ghimj[229] = W_127;
        Ghimj[230] = W_129;
        Ghimj[231] = W_133;
        Ghimj[232] = W_137;
        W_59 = Ghimj[309];
        W_60 = Ghimj[310];
        W_92 = Ghimj[311];
        W_120 = Ghimj[312];
        W_133 = Ghimj[313];
        W_135 = Ghimj[314];
        a = - W_59/ Ghimj[306];
        W_59 = -a;
        W_133 = W_133+ a *Ghimj[307];
        W_135 = W_135+ a *Ghimj[308];
        Ghimj[309] = W_59;
        Ghimj[310] = W_60;
        Ghimj[311] = W_92;
        Ghimj[312] = W_120;
        Ghimj[313] = W_133;
        Ghimj[314] = W_135;
        W_61 = Ghimj[351];
        W_70 = Ghimj[352];
        W_84 = Ghimj[353];
        W_87 = Ghimj[354];
        W_126 = Ghimj[355];
        a = - W_61/ Ghimj[315];
        W_61 = -a;
        W_70 = W_70+ a *Ghimj[316];
        W_87 = W_87+ a *Ghimj[317];
        W_126 = W_126+ a *Ghimj[318];
        Ghimj[351] = W_61;
        Ghimj[352] = W_70;
        Ghimj[353] = W_84;
        Ghimj[354] = W_87;
        Ghimj[355] = W_126;
        W_79 = Ghimj[426];
        W_85 = Ghimj[427];
        W_102 = Ghimj[428];
        W_111 = Ghimj[429];
        W_125 = Ghimj[430];
        W_126 = Ghimj[431];
        W_133 = Ghimj[432];
        W_137 = Ghimj[433];
        a = - W_79/ Ghimj[393];
        W_79 = -a;
        W_102 = W_102+ a *Ghimj[394];
        W_126 = W_126+ a *Ghimj[395];
        W_137 = W_137+ a *Ghimj[396];
        Ghimj[426] = W_79;
        Ghimj[427] = W_85;
        Ghimj[428] = W_102;
        Ghimj[429] = W_111;
        Ghimj[430] = W_125;
        Ghimj[431] = W_126;
        Ghimj[432] = W_133;
        Ghimj[433] = W_137;
        W_62 = Ghimj[434];
        W_69 = Ghimj[435];
        W_86 = Ghimj[436];
        W_93 = Ghimj[437];
        W_125 = Ghimj[438];
        W_126 = Ghimj[439];
        W_133 = Ghimj[440];
        W_137 = Ghimj[441];
        a = - W_62/ Ghimj[319];
        W_62 = -a;
        W_93 = W_93+ a *Ghimj[320];
        W_126 = W_126+ a *Ghimj[321];
        W_133 = W_133+ a *Ghimj[322];
        a = - W_69/ Ghimj[347];
        W_69 = -a;
        W_93 = W_93+ a *Ghimj[348];
        W_126 = W_126+ a *Ghimj[349];
        W_137 = W_137+ a *Ghimj[350];
        Ghimj[434] = W_62;
        Ghimj[435] = W_69;
        Ghimj[436] = W_86;
        Ghimj[437] = W_93;
        Ghimj[438] = W_125;
        Ghimj[439] = W_126;
        Ghimj[440] = W_133;
        Ghimj[441] = W_137;
        W_70 = Ghimj[442];
        W_84 = Ghimj[443];
        W_87 = Ghimj[444];
        W_92 = Ghimj[445];
        W_124 = Ghimj[446];
        W_126 = Ghimj[447];
        W_135 = Ghimj[448];
        W_137 = Ghimj[449];
        a = - W_70/ Ghimj[352];
        W_70 = -a;
        W_84 = W_84+ a *Ghimj[353];
        W_87 = W_87+ a *Ghimj[354];
        W_126 = W_126+ a *Ghimj[355];
        a = - W_84/ Ghimj[421];
        W_84 = -a;
        W_92 = W_92+ a *Ghimj[422];
        W_124 = W_124+ a *Ghimj[423];
        W_135 = W_135+ a *Ghimj[424];
        W_137 = W_137+ a *Ghimj[425];
        Ghimj[442] = W_70;
        Ghimj[443] = W_84;
        Ghimj[444] = W_87;
        Ghimj[445] = W_92;
        Ghimj[446] = W_124;
        Ghimj[447] = W_126;
        Ghimj[448] = W_135;
        Ghimj[449] = W_137;
        W_80 = Ghimj[468];
        W_90 = Ghimj[469];
        W_100 = Ghimj[470];
        W_105 = Ghimj[471];
        W_112 = Ghimj[472];
        W_116 = Ghimj[473];
        W_118 = Ghimj[474];
        W_123 = Ghimj[475];
        W_127 = Ghimj[476];
        W_129 = Ghimj[477];
        W_132 = Ghimj[478];
        W_134 = Ghimj[479];
        W_138 = Ghimj[480];
        a = - W_80/ Ghimj[397];
        W_80 = -a;
        W_90 = W_90+ a *Ghimj[398];
        W_112 = W_112+ a *Ghimj[399];
        W_116 = W_116+ a *Ghimj[400];
        W_127 = W_127+ a *Ghimj[401];
        W_129 = W_129+ a *Ghimj[402];
        W_134 = W_134+ a *Ghimj[403];
        W_138 = W_138+ a *Ghimj[404];
        Ghimj[468] = W_80;
        Ghimj[469] = W_90;
        Ghimj[470] = W_100;
        Ghimj[471] = W_105;
        Ghimj[472] = W_112;
        Ghimj[473] = W_116;
        Ghimj[474] = W_118;
        Ghimj[475] = W_123;
        Ghimj[476] = W_127;
        Ghimj[477] = W_129;
        Ghimj[478] = W_132;
        Ghimj[479] = W_134;
        Ghimj[480] = W_138;
        W_47 = Ghimj[487];
        W_84 = Ghimj[488];
        W_92 = Ghimj[489];
        W_124 = Ghimj[490];
        W_126 = Ghimj[491];
        W_133 = Ghimj[492];
        W_135 = Ghimj[493];
        W_137 = Ghimj[494];
        a = - W_47/ Ghimj[276];
        W_47 = -a;
        W_126 = W_126+ a *Ghimj[277];
        a = - W_84/ Ghimj[421];
        W_84 = -a;
        W_92 = W_92+ a *Ghimj[422];
        W_124 = W_124+ a *Ghimj[423];
        W_135 = W_135+ a *Ghimj[424];
        W_137 = W_137+ a *Ghimj[425];
        Ghimj[487] = W_47;
        Ghimj[488] = W_84;
        Ghimj[489] = W_92;
        Ghimj[490] = W_124;
        Ghimj[491] = W_126;
        Ghimj[492] = W_133;
        Ghimj[493] = W_135;
        Ghimj[494] = W_137;
        W_49 = Ghimj[495];
        W_69 = Ghimj[496];
        W_93 = Ghimj[497];
        W_125 = Ghimj[498];
        W_126 = Ghimj[499];
        W_133 = Ghimj[500];
        W_137 = Ghimj[501];
        a = - W_49/ Ghimj[280];
        W_49 = -a;
        W_126 = W_126+ a *Ghimj[281];
        a = - W_69/ Ghimj[347];
        W_69 = -a;
        W_93 = W_93+ a *Ghimj[348];
        W_126 = W_126+ a *Ghimj[349];
        W_137 = W_137+ a *Ghimj[350];
        Ghimj[495] = W_49;
        Ghimj[496] = W_69;
        Ghimj[497] = W_93;
        Ghimj[498] = W_125;
        Ghimj[499] = W_126;
        Ghimj[500] = W_133;
        Ghimj[501] = W_137;
        W_72 = Ghimj[502];
        W_86 = Ghimj[503];
        W_93 = Ghimj[504];
        W_94 = Ghimj[505];
        W_125 = Ghimj[506];
        W_126 = Ghimj[507];
        W_133 = Ghimj[508];
        W_137 = Ghimj[509];
        a = - W_72/ Ghimj[360];
        W_72 = -a;
        W_94 = W_94+ a *Ghimj[361];
        W_126 = W_126+ a *Ghimj[362];
        W_137 = W_137+ a *Ghimj[363];
        a = - W_86/ Ghimj[436];
        W_86 = -a;
        W_93 = W_93+ a *Ghimj[437];
        W_125 = W_125+ a *Ghimj[438];
        W_126 = W_126+ a *Ghimj[439];
        W_133 = W_133+ a *Ghimj[440];
        W_137 = W_137+ a *Ghimj[441];
        a = - W_93/ Ghimj[497];
        W_93 = -a;
        W_125 = W_125+ a *Ghimj[498];
        W_126 = W_126+ a *Ghimj[499];
        W_133 = W_133+ a *Ghimj[500];
        W_137 = W_137+ a *Ghimj[501];
        Ghimj[502] = W_72;
        Ghimj[503] = W_86;
        Ghimj[504] = W_93;
        Ghimj[505] = W_94;
        Ghimj[506] = W_125;
        Ghimj[507] = W_126;
        Ghimj[508] = W_133;
        Ghimj[509] = W_137;
        W_58 = Ghimj[510];
        W_77 = Ghimj[511];
        W_82 = Ghimj[512];
        W_91 = Ghimj[513];
        W_95 = Ghimj[514];
        W_96 = Ghimj[515];
        W_98 = Ghimj[516];
        W_103 = Ghimj[517];
        W_106 = Ghimj[518];
        W_107 = Ghimj[519];
        W_109 = Ghimj[520];
        W_110 = Ghimj[521];
        W_113 = Ghimj[522];
        W_119 = Ghimj[523];
        W_121 = Ghimj[524];
        W_124 = Ghimj[525];
        W_125 = Ghimj[526];
        W_126 = Ghimj[527];
        W_127 = Ghimj[528];
        W_129 = Ghimj[529];
        W_130 = Ghimj[530];
        W_133 = Ghimj[531];
        W_135 = Ghimj[532];
        W_136 = Ghimj[533];
        W_137 = Ghimj[534];
        a = - W_58/ Ghimj[303];
        W_58 = -a;
        W_91 = W_91+ a *Ghimj[304];
        W_126 = W_126+ a *Ghimj[305];
        a = - W_77/ Ghimj[382];
        W_77 = -a;
        W_121 = W_121+ a *Ghimj[383];
        W_126 = W_126+ a *Ghimj[384];
        W_135 = W_135+ a *Ghimj[385];
        a = - W_82/ Ghimj[412];
        W_82 = -a;
        W_113 = W_113+ a *Ghimj[413];
        W_126 = W_126+ a *Ghimj[414];
        W_137 = W_137+ a *Ghimj[415];
        a = - W_91/ Ghimj[481];
        W_91 = -a;
        W_106 = W_106+ a *Ghimj[482];
        W_109 = W_109+ a *Ghimj[483];
        W_126 = W_126+ a *Ghimj[484];
        W_133 = W_133+ a *Ghimj[485];
        W_136 = W_136+ a *Ghimj[486];
        Ghimj[510] = W_58;
        Ghimj[511] = W_77;
        Ghimj[512] = W_82;
        Ghimj[513] = W_91;
        Ghimj[514] = W_95;
        Ghimj[515] = W_96;
        Ghimj[516] = W_98;
        Ghimj[517] = W_103;
        Ghimj[518] = W_106;
        Ghimj[519] = W_107;
        Ghimj[520] = W_109;
        Ghimj[521] = W_110;
        Ghimj[522] = W_113;
        Ghimj[523] = W_119;
        Ghimj[524] = W_121;
        Ghimj[525] = W_124;
        Ghimj[526] = W_125;
        Ghimj[527] = W_126;
        Ghimj[528] = W_127;
        Ghimj[529] = W_129;
        Ghimj[530] = W_130;
        Ghimj[531] = W_133;
        Ghimj[532] = W_135;
        Ghimj[533] = W_136;
        Ghimj[534] = W_137;
        W_72 = Ghimj[535];
        W_82 = Ghimj[536];
        W_94 = Ghimj[537];
        W_96 = Ghimj[538];
        W_107 = Ghimj[539];
        W_108 = Ghimj[540];
        W_109 = Ghimj[541];
        W_110 = Ghimj[542];
        W_113 = Ghimj[543];
        W_124 = Ghimj[544];
        W_125 = Ghimj[545];
        W_126 = Ghimj[546];
        W_133 = Ghimj[547];
        W_137 = Ghimj[548];
        a = - W_72/ Ghimj[360];
        W_72 = -a;
        W_94 = W_94+ a *Ghimj[361];
        W_126 = W_126+ a *Ghimj[362];
        W_137 = W_137+ a *Ghimj[363];
        a = - W_82/ Ghimj[412];
        W_82 = -a;
        W_113 = W_113+ a *Ghimj[413];
        W_126 = W_126+ a *Ghimj[414];
        W_137 = W_137+ a *Ghimj[415];
        a = - W_94/ Ghimj[505];
        W_94 = -a;
        W_125 = W_125+ a *Ghimj[506];
        W_126 = W_126+ a *Ghimj[507];
        W_133 = W_133+ a *Ghimj[508];
        W_137 = W_137+ a *Ghimj[509];
        Ghimj[535] = W_72;
        Ghimj[536] = W_82;
        Ghimj[537] = W_94;
        Ghimj[538] = W_96;
        Ghimj[539] = W_107;
        Ghimj[540] = W_108;
        Ghimj[541] = W_109;
        Ghimj[542] = W_110;
        Ghimj[543] = W_113;
        Ghimj[544] = W_124;
        Ghimj[545] = W_125;
        Ghimj[546] = W_126;
        Ghimj[547] = W_133;
        Ghimj[548] = W_137;
        W_68 = Ghimj[563];
        W_85 = Ghimj[564];
        W_99 = Ghimj[565];
        W_102 = Ghimj[566];
        W_111 = Ghimj[567];
        W_125 = Ghimj[568];
        W_126 = Ghimj[569];
        W_133 = Ghimj[570];
        W_137 = Ghimj[571];
        a = - W_68/ Ghimj[343];
        W_68 = -a;
        W_99 = W_99+ a *Ghimj[344];
        W_126 = W_126+ a *Ghimj[345];
        W_137 = W_137+ a *Ghimj[346];
        a = - W_85/ Ghimj[427];
        W_85 = -a;
        W_102 = W_102+ a *Ghimj[428];
        W_111 = W_111+ a *Ghimj[429];
        W_125 = W_125+ a *Ghimj[430];
        W_126 = W_126+ a *Ghimj[431];
        W_133 = W_133+ a *Ghimj[432];
        W_137 = W_137+ a *Ghimj[433];
        Ghimj[563] = W_68;
        Ghimj[564] = W_85;
        Ghimj[565] = W_99;
        Ghimj[566] = W_102;
        Ghimj[567] = W_111;
        Ghimj[568] = W_125;
        Ghimj[569] = W_126;
        Ghimj[570] = W_133;
        Ghimj[571] = W_137;
        W_90 = Ghimj[572];
        W_100 = Ghimj[573];
        W_105 = Ghimj[574];
        W_112 = Ghimj[575];
        W_116 = Ghimj[576];
        W_118 = Ghimj[577];
        W_123 = Ghimj[578];
        W_126 = Ghimj[579];
        W_127 = Ghimj[580];
        W_129 = Ghimj[581];
        W_132 = Ghimj[582];
        W_134 = Ghimj[583];
        W_138 = Ghimj[584];
        a = - W_90/ Ghimj[469];
        W_90 = -a;
        W_100 = W_100+ a *Ghimj[470];
        W_105 = W_105+ a *Ghimj[471];
        W_112 = W_112+ a *Ghimj[472];
        W_116 = W_116+ a *Ghimj[473];
        W_118 = W_118+ a *Ghimj[474];
        W_123 = W_123+ a *Ghimj[475];
        W_127 = W_127+ a *Ghimj[476];
        W_129 = W_129+ a *Ghimj[477];
        W_132 = W_132+ a *Ghimj[478];
        W_134 = W_134+ a *Ghimj[479];
        W_138 = W_138+ a *Ghimj[480];
        Ghimj[572] = W_90;
        Ghimj[573] = W_100;
        Ghimj[574] = W_105;
        Ghimj[575] = W_112;
        Ghimj[576] = W_116;
        Ghimj[577] = W_118;
        Ghimj[578] = W_123;
        Ghimj[579] = W_126;
        Ghimj[580] = W_127;
        Ghimj[581] = W_129;
        Ghimj[582] = W_132;
        Ghimj[583] = W_134;
        Ghimj[584] = W_138;
        W_83 = Ghimj[585];
        W_101 = Ghimj[586];
        W_105 = Ghimj[587];
        W_114 = Ghimj[588];
        W_116 = Ghimj[589];
        W_119 = Ghimj[590];
        W_123 = Ghimj[591];
        W_126 = Ghimj[592];
        W_128 = Ghimj[593];
        W_130 = Ghimj[594];
        W_135 = Ghimj[595];
        W_136 = Ghimj[596];
        W_138 = Ghimj[597];
        a = - W_83/ Ghimj[416];
        W_83 = -a;
        W_128 = W_128+ a *Ghimj[417];
        W_135 = W_135+ a *Ghimj[418];
        W_136 = W_136+ a *Ghimj[419];
        W_138 = W_138+ a *Ghimj[420];
        Ghimj[585] = W_83;
        Ghimj[586] = W_101;
        Ghimj[587] = W_105;
        Ghimj[588] = W_114;
        Ghimj[589] = W_116;
        Ghimj[590] = W_119;
        Ghimj[591] = W_123;
        Ghimj[592] = W_126;
        Ghimj[593] = W_128;
        Ghimj[594] = W_130;
        Ghimj[595] = W_135;
        Ghimj[596] = W_136;
        Ghimj[597] = W_138;
        W_40 = Ghimj[598];
        W_79 = Ghimj[599];
        W_102 = Ghimj[600];
        W_125 = Ghimj[601];
        W_126 = Ghimj[602];
        W_133 = Ghimj[603];
        W_137 = Ghimj[604];
        a = - W_40/ Ghimj[260];
        W_40 = -a;
        W_126 = W_126+ a *Ghimj[261];
        a = - W_79/ Ghimj[393];
        W_79 = -a;
        W_102 = W_102+ a *Ghimj[394];
        W_126 = W_126+ a *Ghimj[395];
        W_137 = W_137+ a *Ghimj[396];
        Ghimj[598] = W_40;
        Ghimj[599] = W_79;
        Ghimj[600] = W_102;
        Ghimj[601] = W_125;
        Ghimj[602] = W_126;
        Ghimj[603] = W_133;
        Ghimj[604] = W_137;
        W_64 = Ghimj[630];
        W_67 = Ghimj[631];
        W_82 = Ghimj[632];
        W_91 = Ghimj[633];
        W_94 = Ghimj[634];
        W_106 = Ghimj[635];
        W_108 = Ghimj[636];
        W_109 = Ghimj[637];
        W_113 = Ghimj[638];
        W_115 = Ghimj[639];
        W_124 = Ghimj[640];
        W_125 = Ghimj[641];
        W_126 = Ghimj[642];
        W_133 = Ghimj[643];
        W_135 = Ghimj[644];
        W_136 = Ghimj[645];
        W_137 = Ghimj[646];
        a = - W_64/ Ghimj[327];
        W_64 = -a;
        W_113 = W_113+ a *Ghimj[328];
        W_126 = W_126+ a *Ghimj[329];
        W_135 = W_135+ a *Ghimj[330];
        a = - W_67/ Ghimj[339];
        W_67 = -a;
        W_115 = W_115+ a *Ghimj[340];
        W_126 = W_126+ a *Ghimj[341];
        W_137 = W_137+ a *Ghimj[342];
        a = - W_82/ Ghimj[412];
        W_82 = -a;
        W_113 = W_113+ a *Ghimj[413];
        W_126 = W_126+ a *Ghimj[414];
        W_137 = W_137+ a *Ghimj[415];
        a = - W_91/ Ghimj[481];
        W_91 = -a;
        W_106 = W_106+ a *Ghimj[482];
        W_109 = W_109+ a *Ghimj[483];
        W_126 = W_126+ a *Ghimj[484];
        W_133 = W_133+ a *Ghimj[485];
        W_136 = W_136+ a *Ghimj[486];
        a = - W_94/ Ghimj[505];
        W_94 = -a;
        W_125 = W_125+ a *Ghimj[506];
        W_126 = W_126+ a *Ghimj[507];
        W_133 = W_133+ a *Ghimj[508];
        W_137 = W_137+ a *Ghimj[509];
        a = - W_106/ Ghimj[622];
        W_106 = -a;
        W_124 = W_124+ a *Ghimj[623];
        W_126 = W_126+ a *Ghimj[624];
        W_136 = W_136+ a *Ghimj[625];
        Ghimj[630] = W_64;
        Ghimj[631] = W_67;
        Ghimj[632] = W_82;
        Ghimj[633] = W_91;
        Ghimj[634] = W_94;
        Ghimj[635] = W_106;
        Ghimj[636] = W_108;
        Ghimj[637] = W_109;
        Ghimj[638] = W_113;
        Ghimj[639] = W_115;
        Ghimj[640] = W_124;
        Ghimj[641] = W_125;
        Ghimj[642] = W_126;
        Ghimj[643] = W_133;
        Ghimj[644] = W_135;
        Ghimj[645] = W_136;
        Ghimj[646] = W_137;
        W_106 = Ghimj[647];
        W_109 = Ghimj[648];
        W_124 = Ghimj[649];
        W_125 = Ghimj[650];
        W_126 = Ghimj[651];
        W_133 = Ghimj[652];
        W_136 = Ghimj[653];
        W_137 = Ghimj[654];
        a = - W_106/ Ghimj[622];
        W_106 = -a;
        W_124 = W_124+ a *Ghimj[623];
        W_126 = W_126+ a *Ghimj[624];
        W_136 = W_136+ a *Ghimj[625];
        Ghimj[647] = W_106;
        Ghimj[648] = W_109;
        Ghimj[649] = W_124;
        Ghimj[650] = W_125;
        Ghimj[651] = W_126;
        Ghimj[652] = W_133;
        Ghimj[653] = W_136;
        Ghimj[654] = W_137;
        W_66 = Ghimj[655];
        W_91 = Ghimj[656];
        W_106 = Ghimj[657];
        W_109 = Ghimj[658];
        W_110 = Ghimj[659];
        W_124 = Ghimj[660];
        W_125 = Ghimj[661];
        W_126 = Ghimj[662];
        W_133 = Ghimj[663];
        W_136 = Ghimj[664];
        W_137 = Ghimj[665];
        a = - W_66/ Ghimj[335];
        W_66 = -a;
        W_109 = W_109+ a *Ghimj[336];
        W_126 = W_126+ a *Ghimj[337];
        W_137 = W_137+ a *Ghimj[338];
        a = - W_91/ Ghimj[481];
        W_91 = -a;
        W_106 = W_106+ a *Ghimj[482];
        W_109 = W_109+ a *Ghimj[483];
        W_126 = W_126+ a *Ghimj[484];
        W_133 = W_133+ a *Ghimj[485];
        W_136 = W_136+ a *Ghimj[486];
        a = - W_106/ Ghimj[622];
        W_106 = -a;
        W_124 = W_124+ a *Ghimj[623];
        W_126 = W_126+ a *Ghimj[624];
        W_136 = W_136+ a *Ghimj[625];
        a = - W_109/ Ghimj[648];
        W_109 = -a;
        W_124 = W_124+ a *Ghimj[649];
        W_125 = W_125+ a *Ghimj[650];
        W_126 = W_126+ a *Ghimj[651];
        W_133 = W_133+ a *Ghimj[652];
        W_136 = W_136+ a *Ghimj[653];
        W_137 = W_137+ a *Ghimj[654];
        Ghimj[655] = W_66;
        Ghimj[656] = W_91;
        Ghimj[657] = W_106;
        Ghimj[658] = W_109;
        Ghimj[659] = W_110;
        Ghimj[660] = W_124;
        Ghimj[661] = W_125;
        Ghimj[662] = W_126;
        Ghimj[663] = W_133;
        Ghimj[664] = W_136;
        Ghimj[665] = W_137;
        W_99 = Ghimj[666];
        W_102 = Ghimj[667];
        W_107 = Ghimj[668];
        W_111 = Ghimj[669];
        W_115 = Ghimj[670];
        W_124 = Ghimj[671];
        W_125 = Ghimj[672];
        W_126 = Ghimj[673];
        W_133 = Ghimj[674];
        W_136 = Ghimj[675];
        W_137 = Ghimj[676];
        a = - W_99/ Ghimj[565];
        W_99 = -a;
        W_102 = W_102+ a *Ghimj[566];
        W_111 = W_111+ a *Ghimj[567];
        W_125 = W_125+ a *Ghimj[568];
        W_126 = W_126+ a *Ghimj[569];
        W_133 = W_133+ a *Ghimj[570];
        W_137 = W_137+ a *Ghimj[571];
        a = - W_102/ Ghimj[600];
        W_102 = -a;
        W_125 = W_125+ a *Ghimj[601];
        W_126 = W_126+ a *Ghimj[602];
        W_133 = W_133+ a *Ghimj[603];
        W_137 = W_137+ a *Ghimj[604];
        a = - W_107/ Ghimj[626];
        W_107 = -a;
        W_124 = W_124+ a *Ghimj[627];
        W_126 = W_126+ a *Ghimj[628];
        W_136 = W_136+ a *Ghimj[629];
        Ghimj[666] = W_99;
        Ghimj[667] = W_102;
        Ghimj[668] = W_107;
        Ghimj[669] = W_111;
        Ghimj[670] = W_115;
        Ghimj[671] = W_124;
        Ghimj[672] = W_125;
        Ghimj[673] = W_126;
        Ghimj[674] = W_133;
        Ghimj[675] = W_136;
        Ghimj[676] = W_137;
        W_64 = Ghimj[685];
        W_82 = Ghimj[686];
        W_106 = Ghimj[687];
        W_110 = Ghimj[688];
        W_113 = Ghimj[689];
        W_124 = Ghimj[690];
        W_125 = Ghimj[691];
        W_126 = Ghimj[692];
        W_133 = Ghimj[693];
        W_135 = Ghimj[694];
        W_136 = Ghimj[695];
        W_137 = Ghimj[696];
        a = - W_64/ Ghimj[327];
        W_64 = -a;
        W_113 = W_113+ a *Ghimj[328];
        W_126 = W_126+ a *Ghimj[329];
        W_135 = W_135+ a *Ghimj[330];
        a = - W_82/ Ghimj[412];
        W_82 = -a;
        W_113 = W_113+ a *Ghimj[413];
        W_126 = W_126+ a *Ghimj[414];
        W_137 = W_137+ a *Ghimj[415];
        a = - W_106/ Ghimj[622];
        W_106 = -a;
        W_124 = W_124+ a *Ghimj[623];
        W_126 = W_126+ a *Ghimj[624];
        W_136 = W_136+ a *Ghimj[625];
        a = - W_110/ Ghimj[659];
        W_110 = -a;
        W_124 = W_124+ a *Ghimj[660];
        W_125 = W_125+ a *Ghimj[661];
        W_126 = W_126+ a *Ghimj[662];
        W_133 = W_133+ a *Ghimj[663];
        W_136 = W_136+ a *Ghimj[664];
        W_137 = W_137+ a *Ghimj[665];
        Ghimj[685] = W_64;
        Ghimj[686] = W_82;
        Ghimj[687] = W_106;
        Ghimj[688] = W_110;
        Ghimj[689] = W_113;
        Ghimj[690] = W_124;
        Ghimj[691] = W_125;
        Ghimj[692] = W_126;
        Ghimj[693] = W_133;
        Ghimj[694] = W_135;
        Ghimj[695] = W_136;
        Ghimj[696] = W_137;
        W_67 = Ghimj[703];
        W_103 = Ghimj[704];
        W_107 = Ghimj[705];
        W_115 = Ghimj[706];
        W_124 = Ghimj[707];
        W_126 = Ghimj[708];
        W_127 = Ghimj[709];
        W_129 = Ghimj[710];
        W_133 = Ghimj[711];
        W_136 = Ghimj[712];
        W_137 = Ghimj[713];
        a = - W_67/ Ghimj[339];
        W_67 = -a;
        W_115 = W_115+ a *Ghimj[340];
        W_126 = W_126+ a *Ghimj[341];
        W_137 = W_137+ a *Ghimj[342];
        a = - W_103/ Ghimj[605];
        W_103 = -a;
        W_124 = W_124+ a *Ghimj[606];
        W_126 = W_126+ a *Ghimj[607];
        W_127 = W_127+ a *Ghimj[608];
        W_129 = W_129+ a *Ghimj[609];
        a = - W_107/ Ghimj[626];
        W_107 = -a;
        W_124 = W_124+ a *Ghimj[627];
        W_126 = W_126+ a *Ghimj[628];
        W_136 = W_136+ a *Ghimj[629];
        Ghimj[703] = W_67;
        Ghimj[704] = W_103;
        Ghimj[705] = W_107;
        Ghimj[706] = W_115;
        Ghimj[707] = W_124;
        Ghimj[708] = W_126;
        Ghimj[709] = W_127;
        Ghimj[710] = W_129;
        Ghimj[711] = W_133;
        Ghimj[712] = W_136;
        Ghimj[713] = W_137;
        W_48 = Ghimj[722];
        W_49 = Ghimj[723];
        W_71 = Ghimj[724];
        W_79 = Ghimj[725];
        W_85 = Ghimj[726];
        W_102 = Ghimj[727];
        W_107 = Ghimj[728];
        W_111 = Ghimj[729];
        W_115 = Ghimj[730];
        W_117 = Ghimj[731];
        W_121 = Ghimj[732];
        W_124 = Ghimj[733];
        W_125 = Ghimj[734];
        W_126 = Ghimj[735];
        W_127 = Ghimj[736];
        W_129 = Ghimj[737];
        W_133 = Ghimj[738];
        W_136 = Ghimj[739];
        W_137 = Ghimj[740];
        a = - W_48/ Ghimj[278];
        W_48 = -a;
        W_126 = W_126+ a *Ghimj[279];
        a = - W_49/ Ghimj[280];
        W_49 = -a;
        W_126 = W_126+ a *Ghimj[281];
        a = - W_71/ Ghimj[356];
        W_71 = -a;
        W_117 = W_117+ a *Ghimj[357];
        W_126 = W_126+ a *Ghimj[358];
        W_137 = W_137+ a *Ghimj[359];
        a = - W_79/ Ghimj[393];
        W_79 = -a;
        W_102 = W_102+ a *Ghimj[394];
        W_126 = W_126+ a *Ghimj[395];
        W_137 = W_137+ a *Ghimj[396];
        a = - W_85/ Ghimj[427];
        W_85 = -a;
        W_102 = W_102+ a *Ghimj[428];
        W_111 = W_111+ a *Ghimj[429];
        W_125 = W_125+ a *Ghimj[430];
        W_126 = W_126+ a *Ghimj[431];
        W_133 = W_133+ a *Ghimj[432];
        W_137 = W_137+ a *Ghimj[433];
        a = - W_102/ Ghimj[600];
        W_102 = -a;
        W_125 = W_125+ a *Ghimj[601];
        W_126 = W_126+ a *Ghimj[602];
        W_133 = W_133+ a *Ghimj[603];
        W_137 = W_137+ a *Ghimj[604];
        a = - W_107/ Ghimj[626];
        W_107 = -a;
        W_124 = W_124+ a *Ghimj[627];
        W_126 = W_126+ a *Ghimj[628];
        W_136 = W_136+ a *Ghimj[629];
        a = - W_111/ Ghimj[669];
        W_111 = -a;
        W_115 = W_115+ a *Ghimj[670];
        W_124 = W_124+ a *Ghimj[671];
        W_125 = W_125+ a *Ghimj[672];
        W_126 = W_126+ a *Ghimj[673];
        W_133 = W_133+ a *Ghimj[674];
        W_136 = W_136+ a *Ghimj[675];
        W_137 = W_137+ a *Ghimj[676];
        a = - W_115/ Ghimj[706];
        W_115 = -a;
        W_124 = W_124+ a *Ghimj[707];
        W_126 = W_126+ a *Ghimj[708];
        W_127 = W_127+ a *Ghimj[709];
        W_129 = W_129+ a *Ghimj[710];
        W_133 = W_133+ a *Ghimj[711];
        W_136 = W_136+ a *Ghimj[712];
        W_137 = W_137+ a *Ghimj[713];
        Ghimj[722] = W_48;
        Ghimj[723] = W_49;
        Ghimj[724] = W_71;
        Ghimj[725] = W_79;
        Ghimj[726] = W_85;
        Ghimj[727] = W_102;
        Ghimj[728] = W_107;
        Ghimj[729] = W_111;
        Ghimj[730] = W_115;
        Ghimj[731] = W_117;
        Ghimj[732] = W_121;
        Ghimj[733] = W_124;
        Ghimj[734] = W_125;
        Ghimj[735] = W_126;
        Ghimj[736] = W_127;
        Ghimj[737] = W_129;
        Ghimj[738] = W_133;
        Ghimj[739] = W_136;
        Ghimj[740] = W_137;
        W_100 = Ghimj[741];
        W_105 = Ghimj[742];
        W_112 = Ghimj[743];
        W_116 = Ghimj[744];
        W_118 = Ghimj[745];
        W_123 = Ghimj[746];
        W_125 = Ghimj[747];
        W_126 = Ghimj[748];
        W_127 = Ghimj[749];
        W_128 = Ghimj[750];
        W_129 = Ghimj[751];
        W_131 = Ghimj[752];
        W_132 = Ghimj[753];
        W_134 = Ghimj[754];
        W_135 = Ghimj[755];
        W_137 = Ghimj[756];
        W_138 = Ghimj[757];
        a = - W_100/ Ghimj[573];
        W_100 = -a;
        W_105 = W_105+ a *Ghimj[574];
        W_112 = W_112+ a *Ghimj[575];
        W_116 = W_116+ a *Ghimj[576];
        W_118 = W_118+ a *Ghimj[577];
        W_123 = W_123+ a *Ghimj[578];
        W_126 = W_126+ a *Ghimj[579];
        W_127 = W_127+ a *Ghimj[580];
        W_129 = W_129+ a *Ghimj[581];
        W_132 = W_132+ a *Ghimj[582];
        W_134 = W_134+ a *Ghimj[583];
        W_138 = W_138+ a *Ghimj[584];
        a = - W_105/ Ghimj[616];
        W_105 = -a;
        W_128 = W_128+ a *Ghimj[617];
        W_129 = W_129+ a *Ghimj[618];
        W_132 = W_132+ a *Ghimj[619];
        W_135 = W_135+ a *Ghimj[620];
        W_138 = W_138+ a *Ghimj[621];
        a = - W_112/ Ghimj[677];
        W_112 = -a;
        W_116 = W_116+ a *Ghimj[678];
        W_123 = W_123+ a *Ghimj[679];
        W_126 = W_126+ a *Ghimj[680];
        W_128 = W_128+ a *Ghimj[681];
        W_134 = W_134+ a *Ghimj[682];
        W_137 = W_137+ a *Ghimj[683];
        W_138 = W_138+ a *Ghimj[684];
        a = - W_116/ Ghimj[714];
        W_116 = -a;
        W_123 = W_123+ a *Ghimj[715];
        W_127 = W_127+ a *Ghimj[716];
        W_128 = W_128+ a *Ghimj[717];
        W_131 = W_131+ a *Ghimj[718];
        W_134 = W_134+ a *Ghimj[719];
        W_135 = W_135+ a *Ghimj[720];
        W_138 = W_138+ a *Ghimj[721];
        Ghimj[741] = W_100;
        Ghimj[742] = W_105;
        Ghimj[743] = W_112;
        Ghimj[744] = W_116;
        Ghimj[745] = W_118;
        Ghimj[746] = W_123;
        Ghimj[747] = W_125;
        Ghimj[748] = W_126;
        Ghimj[749] = W_127;
        Ghimj[750] = W_128;
        Ghimj[751] = W_129;
        Ghimj[752] = W_131;
        Ghimj[753] = W_132;
        Ghimj[754] = W_134;
        Ghimj[755] = W_135;
        Ghimj[756] = W_137;
        Ghimj[757] = W_138;
        W_68 = Ghimj[758];
        W_71 = Ghimj[759];
        W_79 = Ghimj[760];
        W_99 = Ghimj[761];
        W_102 = Ghimj[762];
        W_107 = Ghimj[763];
        W_111 = Ghimj[764];
        W_115 = Ghimj[765];
        W_117 = Ghimj[766];
        W_119 = Ghimj[767];
        W_121 = Ghimj[768];
        W_124 = Ghimj[769];
        W_125 = Ghimj[770];
        W_126 = Ghimj[771];
        W_127 = Ghimj[772];
        W_129 = Ghimj[773];
        W_133 = Ghimj[774];
        W_136 = Ghimj[775];
        W_137 = Ghimj[776];
        a = - W_68/ Ghimj[343];
        W_68 = -a;
        W_99 = W_99+ a *Ghimj[344];
        W_126 = W_126+ a *Ghimj[345];
        W_137 = W_137+ a *Ghimj[346];
        a = - W_71/ Ghimj[356];
        W_71 = -a;
        W_117 = W_117+ a *Ghimj[357];
        W_126 = W_126+ a *Ghimj[358];
        W_137 = W_137+ a *Ghimj[359];
        a = - W_79/ Ghimj[393];
        W_79 = -a;
        W_102 = W_102+ a *Ghimj[394];
        W_126 = W_126+ a *Ghimj[395];
        W_137 = W_137+ a *Ghimj[396];
        a = - W_99/ Ghimj[565];
        W_99 = -a;
        W_102 = W_102+ a *Ghimj[566];
        W_111 = W_111+ a *Ghimj[567];
        W_125 = W_125+ a *Ghimj[568];
        W_126 = W_126+ a *Ghimj[569];
        W_133 = W_133+ a *Ghimj[570];
        W_137 = W_137+ a *Ghimj[571];
        a = - W_102/ Ghimj[600];
        W_102 = -a;
        W_125 = W_125+ a *Ghimj[601];
        W_126 = W_126+ a *Ghimj[602];
        W_133 = W_133+ a *Ghimj[603];
        W_137 = W_137+ a *Ghimj[604];
        a = - W_107/ Ghimj[626];
        W_107 = -a;
        W_124 = W_124+ a *Ghimj[627];
        W_126 = W_126+ a *Ghimj[628];
        W_136 = W_136+ a *Ghimj[629];
        a = - W_111/ Ghimj[669];
        W_111 = -a;
        W_115 = W_115+ a *Ghimj[670];
        W_124 = W_124+ a *Ghimj[671];
        W_125 = W_125+ a *Ghimj[672];
        W_126 = W_126+ a *Ghimj[673];
        W_133 = W_133+ a *Ghimj[674];
        W_136 = W_136+ a *Ghimj[675];
        W_137 = W_137+ a *Ghimj[676];
        a = - W_115/ Ghimj[706];
        W_115 = -a;
        W_124 = W_124+ a *Ghimj[707];
        W_126 = W_126+ a *Ghimj[708];
        W_127 = W_127+ a *Ghimj[709];
        W_129 = W_129+ a *Ghimj[710];
        W_133 = W_133+ a *Ghimj[711];
        W_136 = W_136+ a *Ghimj[712];
        W_137 = W_137+ a *Ghimj[713];
        a = - W_117/ Ghimj[731];
        W_117 = -a;
        W_121 = W_121+ a *Ghimj[732];
        W_124 = W_124+ a *Ghimj[733];
        W_125 = W_125+ a *Ghimj[734];
        W_126 = W_126+ a *Ghimj[735];
        W_127 = W_127+ a *Ghimj[736];
        W_129 = W_129+ a *Ghimj[737];
        W_133 = W_133+ a *Ghimj[738];
        W_136 = W_136+ a *Ghimj[739];
        W_137 = W_137+ a *Ghimj[740];
        Ghimj[758] = W_68;
        Ghimj[759] = W_71;
        Ghimj[760] = W_79;
        Ghimj[761] = W_99;
        Ghimj[762] = W_102;
        Ghimj[763] = W_107;
        Ghimj[764] = W_111;
        Ghimj[765] = W_115;
        Ghimj[766] = W_117;
        Ghimj[767] = W_119;
        Ghimj[768] = W_121;
        Ghimj[769] = W_124;
        Ghimj[770] = W_125;
        Ghimj[771] = W_126;
        Ghimj[772] = W_127;
        Ghimj[773] = W_129;
        Ghimj[774] = W_133;
        Ghimj[775] = W_136;
        Ghimj[776] = W_137;
        W_41 = Ghimj[777];
        W_42 = Ghimj[778];
        W_43 = Ghimj[779];
        W_57 = Ghimj[780];
        W_60 = Ghimj[781];
        W_75 = Ghimj[782];
        W_92 = Ghimj[783];
        W_97 = Ghimj[784];
        W_98 = Ghimj[785];
        W_107 = Ghimj[786];
        W_120 = Ghimj[787];
        W_122 = Ghimj[788];
        W_124 = Ghimj[789];
        W_126 = Ghimj[790];
        W_127 = Ghimj[791];
        W_128 = Ghimj[792];
        W_130 = Ghimj[793];
        W_133 = Ghimj[794];
        W_135 = Ghimj[795];
        W_136 = Ghimj[796];
        W_137 = Ghimj[797];
        a = - W_41/ Ghimj[262];
        W_41 = -a;
        W_120 = W_120+ a *Ghimj[263];
        a = - W_42/ Ghimj[264];
        W_42 = -a;
        W_120 = W_120+ a *Ghimj[265];
        a = - W_43/ Ghimj[266];
        W_43 = -a;
        W_120 = W_120+ a *Ghimj[267];
        a = - W_57/ Ghimj[300];
        W_57 = -a;
        W_120 = W_120+ a *Ghimj[301];
        W_126 = W_126+ a *Ghimj[302];
        a = - W_60/ Ghimj[310];
        W_60 = -a;
        W_92 = W_92+ a *Ghimj[311];
        W_120 = W_120+ a *Ghimj[312];
        W_133 = W_133+ a *Ghimj[313];
        W_135 = W_135+ a *Ghimj[314];
        a = - W_75/ Ghimj[374];
        W_75 = -a;
        W_120 = W_120+ a *Ghimj[375];
        W_126 = W_126+ a *Ghimj[376];
        a = - W_92/ Ghimj[489];
        W_92 = -a;
        W_124 = W_124+ a *Ghimj[490];
        W_126 = W_126+ a *Ghimj[491];
        W_133 = W_133+ a *Ghimj[492];
        W_135 = W_135+ a *Ghimj[493];
        W_137 = W_137+ a *Ghimj[494];
        a = - W_97/ Ghimj[549];
        W_97 = -a;
        W_98 = W_98+ a *Ghimj[550];
        W_120 = W_120+ a *Ghimj[551];
        W_122 = W_122+ a *Ghimj[552];
        W_126 = W_126+ a *Ghimj[553];
        W_127 = W_127+ a *Ghimj[554];
        W_130 = W_130+ a *Ghimj[555];
        W_137 = W_137+ a *Ghimj[556];
        a = - W_98/ Ghimj[557];
        W_98 = -a;
        W_107 = W_107+ a *Ghimj[558];
        W_120 = W_120+ a *Ghimj[559];
        W_124 = W_124+ a *Ghimj[560];
        W_126 = W_126+ a *Ghimj[561];
        W_127 = W_127+ a *Ghimj[562];
        a = - W_107/ Ghimj[626];
        W_107 = -a;
        W_124 = W_124+ a *Ghimj[627];
        W_126 = W_126+ a *Ghimj[628];
        W_136 = W_136+ a *Ghimj[629];
        Ghimj[777] = W_41;
        Ghimj[778] = W_42;
        Ghimj[779] = W_43;
        Ghimj[780] = W_57;
        Ghimj[781] = W_60;
        Ghimj[782] = W_75;
        Ghimj[783] = W_92;
        Ghimj[784] = W_97;
        Ghimj[785] = W_98;
        Ghimj[786] = W_107;
        Ghimj[787] = W_120;
        Ghimj[788] = W_122;
        Ghimj[789] = W_124;
        Ghimj[790] = W_126;
        Ghimj[791] = W_127;
        Ghimj[792] = W_128;
        Ghimj[793] = W_130;
        Ghimj[794] = W_133;
        Ghimj[795] = W_135;
        Ghimj[796] = W_136;
        Ghimj[797] = W_137;
        W_38 = Ghimj[798];
        W_63 = Ghimj[799];
        W_68 = Ghimj[800];
        W_72 = Ghimj[801];
        W_77 = Ghimj[802];
        W_82 = Ghimj[803];
        W_85 = Ghimj[804];
        W_86 = Ghimj[805];
        W_93 = Ghimj[806];
        W_94 = Ghimj[807];
        W_96 = Ghimj[808];
        W_99 = Ghimj[809];
        W_102 = Ghimj[810];
        W_106 = Ghimj[811];
        W_107 = Ghimj[812];
        W_108 = Ghimj[813];
        W_109 = Ghimj[814];
        W_110 = Ghimj[815];
        W_111 = Ghimj[816];
        W_113 = Ghimj[817];
        W_115 = Ghimj[818];
        W_117 = Ghimj[819];
        W_119 = Ghimj[820];
        W_121 = Ghimj[821];
        W_124 = Ghimj[822];
        W_125 = Ghimj[823];
        W_126 = Ghimj[824];
        W_127 = Ghimj[825];
        W_129 = Ghimj[826];
        W_133 = Ghimj[827];
        W_135 = Ghimj[828];
        W_136 = Ghimj[829];
        W_137 = Ghimj[830];
        a = - W_38/ Ghimj[255];
        W_38 = -a;
        W_68 = W_68+ a *Ghimj[256];
        W_126 = W_126+ a *Ghimj[257];
        a = - W_63/ Ghimj[323];
        W_63 = -a;
        W_121 = W_121+ a *Ghimj[324];
        W_126 = W_126+ a *Ghimj[325];
        W_137 = W_137+ a *Ghimj[326];
        a = - W_68/ Ghimj[343];
        W_68 = -a;
        W_99 = W_99+ a *Ghimj[344];
        W_126 = W_126+ a *Ghimj[345];
        W_137 = W_137+ a *Ghimj[346];
        a = - W_72/ Ghimj[360];
        W_72 = -a;
        W_94 = W_94+ a *Ghimj[361];
        W_126 = W_126+ a *Ghimj[362];
        W_137 = W_137+ a *Ghimj[363];
        a = - W_77/ Ghimj[382];
        W_77 = -a;
        W_121 = W_121+ a *Ghimj[383];
        W_126 = W_126+ a *Ghimj[384];
        W_135 = W_135+ a *Ghimj[385];
        a = - W_82/ Ghimj[412];
        W_82 = -a;
        W_113 = W_113+ a *Ghimj[413];
        W_126 = W_126+ a *Ghimj[414];
        W_137 = W_137+ a *Ghimj[415];
        a = - W_85/ Ghimj[427];
        W_85 = -a;
        W_102 = W_102+ a *Ghimj[428];
        W_111 = W_111+ a *Ghimj[429];
        W_125 = W_125+ a *Ghimj[430];
        W_126 = W_126+ a *Ghimj[431];
        W_133 = W_133+ a *Ghimj[432];
        W_137 = W_137+ a *Ghimj[433];
        a = - W_86/ Ghimj[436];
        W_86 = -a;
        W_93 = W_93+ a *Ghimj[437];
        W_125 = W_125+ a *Ghimj[438];
        W_126 = W_126+ a *Ghimj[439];
        W_133 = W_133+ a *Ghimj[440];
        W_137 = W_137+ a *Ghimj[441];
        a = - W_93/ Ghimj[497];
        W_93 = -a;
        W_125 = W_125+ a *Ghimj[498];
        W_126 = W_126+ a *Ghimj[499];
        W_133 = W_133+ a *Ghimj[500];
        W_137 = W_137+ a *Ghimj[501];
        a = - W_94/ Ghimj[505];
        W_94 = -a;
        W_125 = W_125+ a *Ghimj[506];
        W_126 = W_126+ a *Ghimj[507];
        W_133 = W_133+ a *Ghimj[508];
        W_137 = W_137+ a *Ghimj[509];
        a = - W_96/ Ghimj[538];
        W_96 = -a;
        W_107 = W_107+ a *Ghimj[539];
        W_108 = W_108+ a *Ghimj[540];
        W_109 = W_109+ a *Ghimj[541];
        W_110 = W_110+ a *Ghimj[542];
        W_113 = W_113+ a *Ghimj[543];
        W_124 = W_124+ a *Ghimj[544];
        W_125 = W_125+ a *Ghimj[545];
        W_126 = W_126+ a *Ghimj[546];
        W_133 = W_133+ a *Ghimj[547];
        W_137 = W_137+ a *Ghimj[548];
        a = - W_99/ Ghimj[565];
        W_99 = -a;
        W_102 = W_102+ a *Ghimj[566];
        W_111 = W_111+ a *Ghimj[567];
        W_125 = W_125+ a *Ghimj[568];
        W_126 = W_126+ a *Ghimj[569];
        W_133 = W_133+ a *Ghimj[570];
        W_137 = W_137+ a *Ghimj[571];
        a = - W_102/ Ghimj[600];
        W_102 = -a;
        W_125 = W_125+ a *Ghimj[601];
        W_126 = W_126+ a *Ghimj[602];
        W_133 = W_133+ a *Ghimj[603];
        W_137 = W_137+ a *Ghimj[604];
        a = - W_106/ Ghimj[622];
        W_106 = -a;
        W_124 = W_124+ a *Ghimj[623];
        W_126 = W_126+ a *Ghimj[624];
        W_136 = W_136+ a *Ghimj[625];
        a = - W_107/ Ghimj[626];
        W_107 = -a;
        W_124 = W_124+ a *Ghimj[627];
        W_126 = W_126+ a *Ghimj[628];
        W_136 = W_136+ a *Ghimj[629];
        a = - W_108/ Ghimj[636];
        W_108 = -a;
        W_109 = W_109+ a *Ghimj[637];
        W_113 = W_113+ a *Ghimj[638];
        W_115 = W_115+ a *Ghimj[639];
        W_124 = W_124+ a *Ghimj[640];
        W_125 = W_125+ a *Ghimj[641];
        W_126 = W_126+ a *Ghimj[642];
        W_133 = W_133+ a *Ghimj[643];
        W_135 = W_135+ a *Ghimj[644];
        W_136 = W_136+ a *Ghimj[645];
        W_137 = W_137+ a *Ghimj[646];
        a = - W_109/ Ghimj[648];
        W_109 = -a;
        W_124 = W_124+ a *Ghimj[649];
        W_125 = W_125+ a *Ghimj[650];
        W_126 = W_126+ a *Ghimj[651];
        W_133 = W_133+ a *Ghimj[652];
        W_136 = W_136+ a *Ghimj[653];
        W_137 = W_137+ a *Ghimj[654];
        a = - W_110/ Ghimj[659];
        W_110 = -a;
        W_124 = W_124+ a *Ghimj[660];
        W_125 = W_125+ a *Ghimj[661];
        W_126 = W_126+ a *Ghimj[662];
        W_133 = W_133+ a *Ghimj[663];
        W_136 = W_136+ a *Ghimj[664];
        W_137 = W_137+ a *Ghimj[665];
        a = - W_111/ Ghimj[669];
        W_111 = -a;
        W_115 = W_115+ a *Ghimj[670];
        W_124 = W_124+ a *Ghimj[671];
        W_125 = W_125+ a *Ghimj[672];
        W_126 = W_126+ a *Ghimj[673];
        W_133 = W_133+ a *Ghimj[674];
        W_136 = W_136+ a *Ghimj[675];
        W_137 = W_137+ a *Ghimj[676];
        a = - W_113/ Ghimj[689];
        W_113 = -a;
        W_124 = W_124+ a *Ghimj[690];
        W_125 = W_125+ a *Ghimj[691];
        W_126 = W_126+ a *Ghimj[692];
        W_133 = W_133+ a *Ghimj[693];
        W_135 = W_135+ a *Ghimj[694];
        W_136 = W_136+ a *Ghimj[695];
        W_137 = W_137+ a *Ghimj[696];
        a = - W_115/ Ghimj[706];
        W_115 = -a;
        W_124 = W_124+ a *Ghimj[707];
        W_126 = W_126+ a *Ghimj[708];
        W_127 = W_127+ a *Ghimj[709];
        W_129 = W_129+ a *Ghimj[710];
        W_133 = W_133+ a *Ghimj[711];
        W_136 = W_136+ a *Ghimj[712];
        W_137 = W_137+ a *Ghimj[713];
        a = - W_117/ Ghimj[731];
        W_117 = -a;
        W_121 = W_121+ a *Ghimj[732];
        W_124 = W_124+ a *Ghimj[733];
        W_125 = W_125+ a *Ghimj[734];
        W_126 = W_126+ a *Ghimj[735];
        W_127 = W_127+ a *Ghimj[736];
        W_129 = W_129+ a *Ghimj[737];
        W_133 = W_133+ a *Ghimj[738];
        W_136 = W_136+ a *Ghimj[739];
        W_137 = W_137+ a *Ghimj[740];
        a = - W_119/ Ghimj[767];
        W_119 = -a;
        W_121 = W_121+ a *Ghimj[768];
        W_124 = W_124+ a *Ghimj[769];
        W_125 = W_125+ a *Ghimj[770];
        W_126 = W_126+ a *Ghimj[771];
        W_127 = W_127+ a *Ghimj[772];
        W_129 = W_129+ a *Ghimj[773];
        W_133 = W_133+ a *Ghimj[774];
        W_136 = W_136+ a *Ghimj[775];
        W_137 = W_137+ a *Ghimj[776];
        Ghimj[798] = W_38;
        Ghimj[799] = W_63;
        Ghimj[800] = W_68;
        Ghimj[801] = W_72;
        Ghimj[802] = W_77;
        Ghimj[803] = W_82;
        Ghimj[804] = W_85;
        Ghimj[805] = W_86;
        Ghimj[806] = W_93;
        Ghimj[807] = W_94;
        Ghimj[808] = W_96;
        Ghimj[809] = W_99;
        Ghimj[810] = W_102;
        Ghimj[811] = W_106;
        Ghimj[812] = W_107;
        Ghimj[813] = W_108;
        Ghimj[814] = W_109;
        Ghimj[815] = W_110;
        Ghimj[816] = W_111;
        Ghimj[817] = W_113;
        Ghimj[818] = W_115;
        Ghimj[819] = W_117;
        Ghimj[820] = W_119;
        Ghimj[821] = W_121;
        Ghimj[822] = W_124;
        Ghimj[823] = W_125;
        Ghimj[824] = W_126;
        Ghimj[825] = W_127;
        Ghimj[826] = W_129;
        Ghimj[827] = W_133;
        Ghimj[828] = W_135;
        Ghimj[829] = W_136;
        Ghimj[830] = W_137;
        W_75 = Ghimj[831];
        W_95 = Ghimj[832];
        W_96 = Ghimj[833];
        W_97 = Ghimj[834];
        W_98 = Ghimj[835];
        W_103 = Ghimj[836];
        W_106 = Ghimj[837];
        W_107 = Ghimj[838];
        W_108 = Ghimj[839];
        W_109 = Ghimj[840];
        W_110 = Ghimj[841];
        W_113 = Ghimj[842];
        W_115 = Ghimj[843];
        W_119 = Ghimj[844];
        W_120 = Ghimj[845];
        W_121 = Ghimj[846];
        W_122 = Ghimj[847];
        W_124 = Ghimj[848];
        W_125 = Ghimj[849];
        W_126 = Ghimj[850];
        W_127 = Ghimj[851];
        W_128 = Ghimj[852];
        W_129 = Ghimj[853];
        W_130 = Ghimj[854];
        W_131 = Ghimj[855];
        W_133 = Ghimj[856];
        W_135 = Ghimj[857];
        W_136 = Ghimj[858];
        W_137 = Ghimj[859];
        W_138 = Ghimj[860];
        a = - W_75/ Ghimj[374];
        W_75 = -a;
        W_120 = W_120+ a *Ghimj[375];
        W_126 = W_126+ a *Ghimj[376];
        a = - W_95/ Ghimj[514];
        W_95 = -a;
        W_96 = W_96+ a *Ghimj[515];
        W_98 = W_98+ a *Ghimj[516];
        W_103 = W_103+ a *Ghimj[517];
        W_106 = W_106+ a *Ghimj[518];
        W_107 = W_107+ a *Ghimj[519];
        W_109 = W_109+ a *Ghimj[520];
        W_110 = W_110+ a *Ghimj[521];
        W_113 = W_113+ a *Ghimj[522];
        W_119 = W_119+ a *Ghimj[523];
        W_121 = W_121+ a *Ghimj[524];
        W_124 = W_124+ a *Ghimj[525];
        W_125 = W_125+ a *Ghimj[526];
        W_126 = W_126+ a *Ghimj[527];
        W_127 = W_127+ a *Ghimj[528];
        W_129 = W_129+ a *Ghimj[529];
        W_130 = W_130+ a *Ghimj[530];
        W_133 = W_133+ a *Ghimj[531];
        W_135 = W_135+ a *Ghimj[532];
        W_136 = W_136+ a *Ghimj[533];
        W_137 = W_137+ a *Ghimj[534];
        a = - W_96/ Ghimj[538];
        W_96 = -a;
        W_107 = W_107+ a *Ghimj[539];
        W_108 = W_108+ a *Ghimj[540];
        W_109 = W_109+ a *Ghimj[541];
        W_110 = W_110+ a *Ghimj[542];
        W_113 = W_113+ a *Ghimj[543];
        W_124 = W_124+ a *Ghimj[544];
        W_125 = W_125+ a *Ghimj[545];
        W_126 = W_126+ a *Ghimj[546];
        W_133 = W_133+ a *Ghimj[547];
        W_137 = W_137+ a *Ghimj[548];
        a = - W_97/ Ghimj[549];
        W_97 = -a;
        W_98 = W_98+ a *Ghimj[550];
        W_120 = W_120+ a *Ghimj[551];
        W_122 = W_122+ a *Ghimj[552];
        W_126 = W_126+ a *Ghimj[553];
        W_127 = W_127+ a *Ghimj[554];
        W_130 = W_130+ a *Ghimj[555];
        W_137 = W_137+ a *Ghimj[556];
        a = - W_98/ Ghimj[557];
        W_98 = -a;
        W_107 = W_107+ a *Ghimj[558];
        W_120 = W_120+ a *Ghimj[559];
        W_124 = W_124+ a *Ghimj[560];
        W_126 = W_126+ a *Ghimj[561];
        W_127 = W_127+ a *Ghimj[562];
        a = - W_103/ Ghimj[605];
        W_103 = -a;
        W_124 = W_124+ a *Ghimj[606];
        W_126 = W_126+ a *Ghimj[607];
        W_127 = W_127+ a *Ghimj[608];
        W_129 = W_129+ a *Ghimj[609];
        a = - W_106/ Ghimj[622];
        W_106 = -a;
        W_124 = W_124+ a *Ghimj[623];
        W_126 = W_126+ a *Ghimj[624];
        W_136 = W_136+ a *Ghimj[625];
        a = - W_107/ Ghimj[626];
        W_107 = -a;
        W_124 = W_124+ a *Ghimj[627];
        W_126 = W_126+ a *Ghimj[628];
        W_136 = W_136+ a *Ghimj[629];
        a = - W_108/ Ghimj[636];
        W_108 = -a;
        W_109 = W_109+ a *Ghimj[637];
        W_113 = W_113+ a *Ghimj[638];
        W_115 = W_115+ a *Ghimj[639];
        W_124 = W_124+ a *Ghimj[640];
        W_125 = W_125+ a *Ghimj[641];
        W_126 = W_126+ a *Ghimj[642];
        W_133 = W_133+ a *Ghimj[643];
        W_135 = W_135+ a *Ghimj[644];
        W_136 = W_136+ a *Ghimj[645];
        W_137 = W_137+ a *Ghimj[646];
        a = - W_109/ Ghimj[648];
        W_109 = -a;
        W_124 = W_124+ a *Ghimj[649];
        W_125 = W_125+ a *Ghimj[650];
        W_126 = W_126+ a *Ghimj[651];
        W_133 = W_133+ a *Ghimj[652];
        W_136 = W_136+ a *Ghimj[653];
        W_137 = W_137+ a *Ghimj[654];
        a = - W_110/ Ghimj[659];
        W_110 = -a;
        W_124 = W_124+ a *Ghimj[660];
        W_125 = W_125+ a *Ghimj[661];
        W_126 = W_126+ a *Ghimj[662];
        W_133 = W_133+ a *Ghimj[663];
        W_136 = W_136+ a *Ghimj[664];
        W_137 = W_137+ a *Ghimj[665];
        a = - W_113/ Ghimj[689];
        W_113 = -a;
        W_124 = W_124+ a *Ghimj[690];
        W_125 = W_125+ a *Ghimj[691];
        W_126 = W_126+ a *Ghimj[692];
        W_133 = W_133+ a *Ghimj[693];
        W_135 = W_135+ a *Ghimj[694];
        W_136 = W_136+ a *Ghimj[695];
        W_137 = W_137+ a *Ghimj[696];
        a = - W_115/ Ghimj[706];
        W_115 = -a;
        W_124 = W_124+ a *Ghimj[707];
        W_126 = W_126+ a *Ghimj[708];
        W_127 = W_127+ a *Ghimj[709];
        W_129 = W_129+ a *Ghimj[710];
        W_133 = W_133+ a *Ghimj[711];
        W_136 = W_136+ a *Ghimj[712];
        W_137 = W_137+ a *Ghimj[713];
        a = - W_119/ Ghimj[767];
        W_119 = -a;
        W_121 = W_121+ a *Ghimj[768];
        W_124 = W_124+ a *Ghimj[769];
        W_125 = W_125+ a *Ghimj[770];
        W_126 = W_126+ a *Ghimj[771];
        W_127 = W_127+ a *Ghimj[772];
        W_129 = W_129+ a *Ghimj[773];
        W_133 = W_133+ a *Ghimj[774];
        W_136 = W_136+ a *Ghimj[775];
        W_137 = W_137+ a *Ghimj[776];
        a = - W_120/ Ghimj[787];
        W_120 = -a;
        W_122 = W_122+ a *Ghimj[788];
        W_124 = W_124+ a *Ghimj[789];
        W_126 = W_126+ a *Ghimj[790];
        W_127 = W_127+ a *Ghimj[791];
        W_128 = W_128+ a *Ghimj[792];
        W_130 = W_130+ a *Ghimj[793];
        W_133 = W_133+ a *Ghimj[794];
        W_135 = W_135+ a *Ghimj[795];
        W_136 = W_136+ a *Ghimj[796];
        W_137 = W_137+ a *Ghimj[797];
        a = - W_121/ Ghimj[821];
        W_121 = -a;
        W_124 = W_124+ a *Ghimj[822];
        W_125 = W_125+ a *Ghimj[823];
        W_126 = W_126+ a *Ghimj[824];
        W_127 = W_127+ a *Ghimj[825];
        W_129 = W_129+ a *Ghimj[826];
        W_133 = W_133+ a *Ghimj[827];
        W_135 = W_135+ a *Ghimj[828];
        W_136 = W_136+ a *Ghimj[829];
        W_137 = W_137+ a *Ghimj[830];
        Ghimj[831] = W_75;
        Ghimj[832] = W_95;
        Ghimj[833] = W_96;
        Ghimj[834] = W_97;
        Ghimj[835] = W_98;
        Ghimj[836] = W_103;
        Ghimj[837] = W_106;
        Ghimj[838] = W_107;
        Ghimj[839] = W_108;
        Ghimj[840] = W_109;
        Ghimj[841] = W_110;
        Ghimj[842] = W_113;
        Ghimj[843] = W_115;
        Ghimj[844] = W_119;
        Ghimj[845] = W_120;
        Ghimj[846] = W_121;
        Ghimj[847] = W_122;
        Ghimj[848] = W_124;
        Ghimj[849] = W_125;
        Ghimj[850] = W_126;
        Ghimj[851] = W_127;
        Ghimj[852] = W_128;
        Ghimj[853] = W_129;
        Ghimj[854] = W_130;
        Ghimj[855] = W_131;
        Ghimj[856] = W_133;
        Ghimj[857] = W_135;
        Ghimj[858] = W_136;
        Ghimj[859] = W_137;
        Ghimj[860] = W_138;
        W_103 = Ghimj[861];
        W_104 = Ghimj[862];
        W_112 = Ghimj[863];
        W_114 = Ghimj[864];
        W_116 = Ghimj[865];
        W_118 = Ghimj[866];
        W_119 = Ghimj[867];
        W_121 = Ghimj[868];
        W_123 = Ghimj[869];
        W_124 = Ghimj[870];
        W_125 = Ghimj[871];
        W_126 = Ghimj[872];
        W_127 = Ghimj[873];
        W_128 = Ghimj[874];
        W_129 = Ghimj[875];
        W_130 = Ghimj[876];
        W_131 = Ghimj[877];
        W_132 = Ghimj[878];
        W_133 = Ghimj[879];
        W_134 = Ghimj[880];
        W_135 = Ghimj[881];
        W_136 = Ghimj[882];
        W_137 = Ghimj[883];
        W_138 = Ghimj[884];
        a = - W_103/ Ghimj[605];
        W_103 = -a;
        W_124 = W_124+ a *Ghimj[606];
        W_126 = W_126+ a *Ghimj[607];
        W_127 = W_127+ a *Ghimj[608];
        W_129 = W_129+ a *Ghimj[609];
        a = - W_104/ Ghimj[610];
        W_104 = -a;
        W_125 = W_125+ a *Ghimj[611];
        W_126 = W_126+ a *Ghimj[612];
        W_127 = W_127+ a *Ghimj[613];
        W_129 = W_129+ a *Ghimj[614];
        W_137 = W_137+ a *Ghimj[615];
        a = - W_112/ Ghimj[677];
        W_112 = -a;
        W_116 = W_116+ a *Ghimj[678];
        W_123 = W_123+ a *Ghimj[679];
        W_126 = W_126+ a *Ghimj[680];
        W_128 = W_128+ a *Ghimj[681];
        W_134 = W_134+ a *Ghimj[682];
        W_137 = W_137+ a *Ghimj[683];
        W_138 = W_138+ a *Ghimj[684];
        a = - W_114/ Ghimj[697];
        W_114 = -a;
        W_126 = W_126+ a *Ghimj[698];
        W_127 = W_127+ a *Ghimj[699];
        W_129 = W_129+ a *Ghimj[700];
        W_132 = W_132+ a *Ghimj[701];
        W_136 = W_136+ a *Ghimj[702];
        a = - W_116/ Ghimj[714];
        W_116 = -a;
        W_123 = W_123+ a *Ghimj[715];
        W_127 = W_127+ a *Ghimj[716];
        W_128 = W_128+ a *Ghimj[717];
        W_131 = W_131+ a *Ghimj[718];
        W_134 = W_134+ a *Ghimj[719];
        W_135 = W_135+ a *Ghimj[720];
        W_138 = W_138+ a *Ghimj[721];
        a = - W_118/ Ghimj[745];
        W_118 = -a;
        W_123 = W_123+ a *Ghimj[746];
        W_125 = W_125+ a *Ghimj[747];
        W_126 = W_126+ a *Ghimj[748];
        W_127 = W_127+ a *Ghimj[749];
        W_128 = W_128+ a *Ghimj[750];
        W_129 = W_129+ a *Ghimj[751];
        W_131 = W_131+ a *Ghimj[752];
        W_132 = W_132+ a *Ghimj[753];
        W_134 = W_134+ a *Ghimj[754];
        W_135 = W_135+ a *Ghimj[755];
        W_137 = W_137+ a *Ghimj[756];
        W_138 = W_138+ a *Ghimj[757];
        a = - W_119/ Ghimj[767];
        W_119 = -a;
        W_121 = W_121+ a *Ghimj[768];
        W_124 = W_124+ a *Ghimj[769];
        W_125 = W_125+ a *Ghimj[770];
        W_126 = W_126+ a *Ghimj[771];
        W_127 = W_127+ a *Ghimj[772];
        W_129 = W_129+ a *Ghimj[773];
        W_133 = W_133+ a *Ghimj[774];
        W_136 = W_136+ a *Ghimj[775];
        W_137 = W_137+ a *Ghimj[776];
        a = - W_121/ Ghimj[821];
        W_121 = -a;
        W_124 = W_124+ a *Ghimj[822];
        W_125 = W_125+ a *Ghimj[823];
        W_126 = W_126+ a *Ghimj[824];
        W_127 = W_127+ a *Ghimj[825];
        W_129 = W_129+ a *Ghimj[826];
        W_133 = W_133+ a *Ghimj[827];
        W_135 = W_135+ a *Ghimj[828];
        W_136 = W_136+ a *Ghimj[829];
        W_137 = W_137+ a *Ghimj[830];
        Ghimj[861] = W_103;
        Ghimj[862] = W_104;
        Ghimj[863] = W_112;
        Ghimj[864] = W_114;
        Ghimj[865] = W_116;
        Ghimj[866] = W_118;
        Ghimj[867] = W_119;
        Ghimj[868] = W_121;
        Ghimj[869] = W_123;
        Ghimj[870] = W_124;
        Ghimj[871] = W_125;
        Ghimj[872] = W_126;
        Ghimj[873] = W_127;
        Ghimj[874] = W_128;
        Ghimj[875] = W_129;
        Ghimj[876] = W_130;
        Ghimj[877] = W_131;
        Ghimj[878] = W_132;
        Ghimj[879] = W_133;
        Ghimj[880] = W_134;
        Ghimj[881] = W_135;
        Ghimj[882] = W_136;
        Ghimj[883] = W_137;
        Ghimj[884] = W_138;
        W_81 = Ghimj[885];
        W_84 = Ghimj[886];
        W_92 = Ghimj[887];
        W_103 = Ghimj[888];
        W_106 = Ghimj[889];
        W_107 = Ghimj[890];
        W_110 = Ghimj[891];
        W_114 = Ghimj[892];
        W_120 = Ghimj[893];
        W_121 = Ghimj[894];
        W_122 = Ghimj[895];
        W_124 = Ghimj[896];
        W_125 = Ghimj[897];
        W_126 = Ghimj[898];
        W_127 = Ghimj[899];
        W_128 = Ghimj[900];
        W_129 = Ghimj[901];
        W_130 = Ghimj[902];
        W_131 = Ghimj[903];
        W_132 = Ghimj[904];
        W_133 = Ghimj[905];
        W_135 = Ghimj[906];
        W_136 = Ghimj[907];
        W_137 = Ghimj[908];
        W_138 = Ghimj[909];
        a = - W_81/ Ghimj[405];
        W_81 = -a;
        W_114 = W_114+ a *Ghimj[406];
        W_124 = W_124+ a *Ghimj[407];
        W_126 = W_126+ a *Ghimj[408];
        W_127 = W_127+ a *Ghimj[409];
        W_129 = W_129+ a *Ghimj[410];
        W_136 = W_136+ a *Ghimj[411];
        a = - W_84/ Ghimj[421];
        W_84 = -a;
        W_92 = W_92+ a *Ghimj[422];
        W_124 = W_124+ a *Ghimj[423];
        W_135 = W_135+ a *Ghimj[424];
        W_137 = W_137+ a *Ghimj[425];
        a = - W_92/ Ghimj[489];
        W_92 = -a;
        W_124 = W_124+ a *Ghimj[490];
        W_126 = W_126+ a *Ghimj[491];
        W_133 = W_133+ a *Ghimj[492];
        W_135 = W_135+ a *Ghimj[493];
        W_137 = W_137+ a *Ghimj[494];
        a = - W_103/ Ghimj[605];
        W_103 = -a;
        W_124 = W_124+ a *Ghimj[606];
        W_126 = W_126+ a *Ghimj[607];
        W_127 = W_127+ a *Ghimj[608];
        W_129 = W_129+ a *Ghimj[609];
        a = - W_106/ Ghimj[622];
        W_106 = -a;
        W_124 = W_124+ a *Ghimj[623];
        W_126 = W_126+ a *Ghimj[624];
        W_136 = W_136+ a *Ghimj[625];
        a = - W_107/ Ghimj[626];
        W_107 = -a;
        W_124 = W_124+ a *Ghimj[627];
        W_126 = W_126+ a *Ghimj[628];
        W_136 = W_136+ a *Ghimj[629];
        a = - W_110/ Ghimj[659];
        W_110 = -a;
        W_124 = W_124+ a *Ghimj[660];
        W_125 = W_125+ a *Ghimj[661];
        W_126 = W_126+ a *Ghimj[662];
        W_133 = W_133+ a *Ghimj[663];
        W_136 = W_136+ a *Ghimj[664];
        W_137 = W_137+ a *Ghimj[665];
        a = - W_114/ Ghimj[697];
        W_114 = -a;
        W_126 = W_126+ a *Ghimj[698];
        W_127 = W_127+ a *Ghimj[699];
        W_129 = W_129+ a *Ghimj[700];
        W_132 = W_132+ a *Ghimj[701];
        W_136 = W_136+ a *Ghimj[702];
        a = - W_120/ Ghimj[787];
        W_120 = -a;
        W_122 = W_122+ a *Ghimj[788];
        W_124 = W_124+ a *Ghimj[789];
        W_126 = W_126+ a *Ghimj[790];
        W_127 = W_127+ a *Ghimj[791];
        W_128 = W_128+ a *Ghimj[792];
        W_130 = W_130+ a *Ghimj[793];
        W_133 = W_133+ a *Ghimj[794];
        W_135 = W_135+ a *Ghimj[795];
        W_136 = W_136+ a *Ghimj[796];
        W_137 = W_137+ a *Ghimj[797];
        a = - W_121/ Ghimj[821];
        W_121 = -a;
        W_124 = W_124+ a *Ghimj[822];
        W_125 = W_125+ a *Ghimj[823];
        W_126 = W_126+ a *Ghimj[824];
        W_127 = W_127+ a *Ghimj[825];
        W_129 = W_129+ a *Ghimj[826];
        W_133 = W_133+ a *Ghimj[827];
        W_135 = W_135+ a *Ghimj[828];
        W_136 = W_136+ a *Ghimj[829];
        W_137 = W_137+ a *Ghimj[830];
        a = - W_122/ Ghimj[847];
        W_122 = -a;
        W_124 = W_124+ a *Ghimj[848];
        W_125 = W_125+ a *Ghimj[849];
        W_126 = W_126+ a *Ghimj[850];
        W_127 = W_127+ a *Ghimj[851];
        W_128 = W_128+ a *Ghimj[852];
        W_129 = W_129+ a *Ghimj[853];
        W_130 = W_130+ a *Ghimj[854];
        W_131 = W_131+ a *Ghimj[855];
        W_133 = W_133+ a *Ghimj[856];
        W_135 = W_135+ a *Ghimj[857];
        W_136 = W_136+ a *Ghimj[858];
        W_137 = W_137+ a *Ghimj[859];
        W_138 = W_138+ a *Ghimj[860];
        Ghimj[885] = W_81;
        Ghimj[886] = W_84;
        Ghimj[887] = W_92;
        Ghimj[888] = W_103;
        Ghimj[889] = W_106;
        Ghimj[890] = W_107;
        Ghimj[891] = W_110;
        Ghimj[892] = W_114;
        Ghimj[893] = W_120;
        Ghimj[894] = W_121;
        Ghimj[895] = W_122;
        Ghimj[896] = W_124;
        Ghimj[897] = W_125;
        Ghimj[898] = W_126;
        Ghimj[899] = W_127;
        Ghimj[900] = W_128;
        Ghimj[901] = W_129;
        Ghimj[902] = W_130;
        Ghimj[903] = W_131;
        Ghimj[904] = W_132;
        Ghimj[905] = W_133;
        Ghimj[906] = W_135;
        Ghimj[907] = W_136;
        Ghimj[908] = W_137;
        Ghimj[909] = W_138;
        W_3 = Ghimj[910];
        W_53 = Ghimj[911];
        W_63 = Ghimj[912];
        W_65 = Ghimj[913];
        W_74 = Ghimj[914];
        W_75 = Ghimj[915];
        W_81 = Ghimj[916];
        W_86 = Ghimj[917];
        W_93 = Ghimj[918];
        W_94 = Ghimj[919];
        W_98 = Ghimj[920];
        W_102 = Ghimj[921];
        W_104 = Ghimj[922];
        W_106 = Ghimj[923];
        W_107 = Ghimj[924];
        W_109 = Ghimj[925];
        W_113 = Ghimj[926];
        W_114 = Ghimj[927];
        W_117 = Ghimj[928];
        W_119 = Ghimj[929];
        W_120 = Ghimj[930];
        W_121 = Ghimj[931];
        W_122 = Ghimj[932];
        W_124 = Ghimj[933];
        W_125 = Ghimj[934];
        W_126 = Ghimj[935];
        W_127 = Ghimj[936];
        W_128 = Ghimj[937];
        W_129 = Ghimj[938];
        W_130 = Ghimj[939];
        W_131 = Ghimj[940];
        W_132 = Ghimj[941];
        W_133 = Ghimj[942];
        W_134 = Ghimj[943];
        W_135 = Ghimj[944];
        W_136 = Ghimj[945];
        W_137 = Ghimj[946];
        W_138 = Ghimj[947];
        a = - W_3/ Ghimj[3];
        W_3 = -a;
        a = - W_53/ Ghimj[290];
        W_53 = -a;
        W_126 = W_126+ a *Ghimj[291];
        a = - W_63/ Ghimj[323];
        W_63 = -a;
        W_121 = W_121+ a *Ghimj[324];
        W_126 = W_126+ a *Ghimj[325];
        W_137 = W_137+ a *Ghimj[326];
        a = - W_65/ Ghimj[331];
        W_65 = -a;
        W_114 = W_114+ a *Ghimj[332];
        W_126 = W_126+ a *Ghimj[333];
        W_132 = W_132+ a *Ghimj[334];
        a = - W_74/ Ghimj[368];
        W_74 = -a;
        W_117 = W_117+ a *Ghimj[369];
        W_121 = W_121+ a *Ghimj[370];
        W_125 = W_125+ a *Ghimj[371];
        W_126 = W_126+ a *Ghimj[372];
        W_137 = W_137+ a *Ghimj[373];
        a = - W_75/ Ghimj[374];
        W_75 = -a;
        W_120 = W_120+ a *Ghimj[375];
        W_126 = W_126+ a *Ghimj[376];
        a = - W_81/ Ghimj[405];
        W_81 = -a;
        W_114 = W_114+ a *Ghimj[406];
        W_124 = W_124+ a *Ghimj[407];
        W_126 = W_126+ a *Ghimj[408];
        W_127 = W_127+ a *Ghimj[409];
        W_129 = W_129+ a *Ghimj[410];
        W_136 = W_136+ a *Ghimj[411];
        a = - W_86/ Ghimj[436];
        W_86 = -a;
        W_93 = W_93+ a *Ghimj[437];
        W_125 = W_125+ a *Ghimj[438];
        W_126 = W_126+ a *Ghimj[439];
        W_133 = W_133+ a *Ghimj[440];
        W_137 = W_137+ a *Ghimj[441];
        a = - W_93/ Ghimj[497];
        W_93 = -a;
        W_125 = W_125+ a *Ghimj[498];
        W_126 = W_126+ a *Ghimj[499];
        W_133 = W_133+ a *Ghimj[500];
        W_137 = W_137+ a *Ghimj[501];
        a = - W_94/ Ghimj[505];
        W_94 = -a;
        W_125 = W_125+ a *Ghimj[506];
        W_126 = W_126+ a *Ghimj[507];
        W_133 = W_133+ a *Ghimj[508];
        W_137 = W_137+ a *Ghimj[509];
        a = - W_98/ Ghimj[557];
        W_98 = -a;
        W_107 = W_107+ a *Ghimj[558];
        W_120 = W_120+ a *Ghimj[559];
        W_124 = W_124+ a *Ghimj[560];
        W_126 = W_126+ a *Ghimj[561];
        W_127 = W_127+ a *Ghimj[562];
        a = - W_102/ Ghimj[600];
        W_102 = -a;
        W_125 = W_125+ a *Ghimj[601];
        W_126 = W_126+ a *Ghimj[602];
        W_133 = W_133+ a *Ghimj[603];
        W_137 = W_137+ a *Ghimj[604];
        a = - W_104/ Ghimj[610];
        W_104 = -a;
        W_125 = W_125+ a *Ghimj[611];
        W_126 = W_126+ a *Ghimj[612];
        W_127 = W_127+ a *Ghimj[613];
        W_129 = W_129+ a *Ghimj[614];
        W_137 = W_137+ a *Ghimj[615];
        a = - W_106/ Ghimj[622];
        W_106 = -a;
        W_124 = W_124+ a *Ghimj[623];
        W_126 = W_126+ a *Ghimj[624];
        W_136 = W_136+ a *Ghimj[625];
        a = - W_107/ Ghimj[626];
        W_107 = -a;
        W_124 = W_124+ a *Ghimj[627];
        W_126 = W_126+ a *Ghimj[628];
        W_136 = W_136+ a *Ghimj[629];
        a = - W_109/ Ghimj[648];
        W_109 = -a;
        W_124 = W_124+ a *Ghimj[649];
        W_125 = W_125+ a *Ghimj[650];
        W_126 = W_126+ a *Ghimj[651];
        W_133 = W_133+ a *Ghimj[652];
        W_136 = W_136+ a *Ghimj[653];
        W_137 = W_137+ a *Ghimj[654];
        a = - W_113/ Ghimj[689];
        W_113 = -a;
        W_124 = W_124+ a *Ghimj[690];
        W_125 = W_125+ a *Ghimj[691];
        W_126 = W_126+ a *Ghimj[692];
        W_133 = W_133+ a *Ghimj[693];
        W_135 = W_135+ a *Ghimj[694];
        W_136 = W_136+ a *Ghimj[695];
        W_137 = W_137+ a *Ghimj[696];
        a = - W_114/ Ghimj[697];
        W_114 = -a;
        W_126 = W_126+ a *Ghimj[698];
        W_127 = W_127+ a *Ghimj[699];
        W_129 = W_129+ a *Ghimj[700];
        W_132 = W_132+ a *Ghimj[701];
        W_136 = W_136+ a *Ghimj[702];
        a = - W_117/ Ghimj[731];
        W_117 = -a;
        W_121 = W_121+ a *Ghimj[732];
        W_124 = W_124+ a *Ghimj[733];
        W_125 = W_125+ a *Ghimj[734];
        W_126 = W_126+ a *Ghimj[735];
        W_127 = W_127+ a *Ghimj[736];
        W_129 = W_129+ a *Ghimj[737];
        W_133 = W_133+ a *Ghimj[738];
        W_136 = W_136+ a *Ghimj[739];
        W_137 = W_137+ a *Ghimj[740];
        a = - W_119/ Ghimj[767];
        W_119 = -a;
        W_121 = W_121+ a *Ghimj[768];
        W_124 = W_124+ a *Ghimj[769];
        W_125 = W_125+ a *Ghimj[770];
        W_126 = W_126+ a *Ghimj[771];
        W_127 = W_127+ a *Ghimj[772];
        W_129 = W_129+ a *Ghimj[773];
        W_133 = W_133+ a *Ghimj[774];
        W_136 = W_136+ a *Ghimj[775];
        W_137 = W_137+ a *Ghimj[776];
        a = - W_120/ Ghimj[787];
        W_120 = -a;
        W_122 = W_122+ a *Ghimj[788];
        W_124 = W_124+ a *Ghimj[789];
        W_126 = W_126+ a *Ghimj[790];
        W_127 = W_127+ a *Ghimj[791];
        W_128 = W_128+ a *Ghimj[792];
        W_130 = W_130+ a *Ghimj[793];
        W_133 = W_133+ a *Ghimj[794];
        W_135 = W_135+ a *Ghimj[795];
        W_136 = W_136+ a *Ghimj[796];
        W_137 = W_137+ a *Ghimj[797];
        a = - W_121/ Ghimj[821];
        W_121 = -a;
        W_124 = W_124+ a *Ghimj[822];
        W_125 = W_125+ a *Ghimj[823];
        W_126 = W_126+ a *Ghimj[824];
        W_127 = W_127+ a *Ghimj[825];
        W_129 = W_129+ a *Ghimj[826];
        W_133 = W_133+ a *Ghimj[827];
        W_135 = W_135+ a *Ghimj[828];
        W_136 = W_136+ a *Ghimj[829];
        W_137 = W_137+ a *Ghimj[830];
        a = - W_122/ Ghimj[847];
        W_122 = -a;
        W_124 = W_124+ a *Ghimj[848];
        W_125 = W_125+ a *Ghimj[849];
        W_126 = W_126+ a *Ghimj[850];
        W_127 = W_127+ a *Ghimj[851];
        W_128 = W_128+ a *Ghimj[852];
        W_129 = W_129+ a *Ghimj[853];
        W_130 = W_130+ a *Ghimj[854];
        W_131 = W_131+ a *Ghimj[855];
        W_133 = W_133+ a *Ghimj[856];
        W_135 = W_135+ a *Ghimj[857];
        W_136 = W_136+ a *Ghimj[858];
        W_137 = W_137+ a *Ghimj[859];
        W_138 = W_138+ a *Ghimj[860];
        a = - W_124/ Ghimj[896];
        W_124 = -a;
        W_125 = W_125+ a *Ghimj[897];
        W_126 = W_126+ a *Ghimj[898];
        W_127 = W_127+ a *Ghimj[899];
        W_128 = W_128+ a *Ghimj[900];
        W_129 = W_129+ a *Ghimj[901];
        W_130 = W_130+ a *Ghimj[902];
        W_131 = W_131+ a *Ghimj[903];
        W_132 = W_132+ a *Ghimj[904];
        W_133 = W_133+ a *Ghimj[905];
        W_135 = W_135+ a *Ghimj[906];
        W_136 = W_136+ a *Ghimj[907];
        W_137 = W_137+ a *Ghimj[908];
        W_138 = W_138+ a *Ghimj[909];
        Ghimj[910] = W_3;
        Ghimj[911] = W_53;
        Ghimj[912] = W_63;
        Ghimj[913] = W_65;
        Ghimj[914] = W_74;
        Ghimj[915] = W_75;
        Ghimj[916] = W_81;
        Ghimj[917] = W_86;
        Ghimj[918] = W_93;
        Ghimj[919] = W_94;
        Ghimj[920] = W_98;
        Ghimj[921] = W_102;
        Ghimj[922] = W_104;
        Ghimj[923] = W_106;
        Ghimj[924] = W_107;
        Ghimj[925] = W_109;
        Ghimj[926] = W_113;
        Ghimj[927] = W_114;
        Ghimj[928] = W_117;
        Ghimj[929] = W_119;
        Ghimj[930] = W_120;
        Ghimj[931] = W_121;
        Ghimj[932] = W_122;
        Ghimj[933] = W_124;
        Ghimj[934] = W_125;
        Ghimj[935] = W_126;
        Ghimj[936] = W_127;
        Ghimj[937] = W_128;
        Ghimj[938] = W_129;
        Ghimj[939] = W_130;
        Ghimj[940] = W_131;
        Ghimj[941] = W_132;
        Ghimj[942] = W_133;
        Ghimj[943] = W_134;
        Ghimj[944] = W_135;
        Ghimj[945] = W_136;
        Ghimj[946] = W_137;
        Ghimj[947] = W_138;
        W_40 = Ghimj[948];
        W_44 = Ghimj[949];
        W_45 = Ghimj[950];
        W_47 = Ghimj[951];
        W_48 = Ghimj[952];
        W_49 = Ghimj[953];
        W_52 = Ghimj[954];
        W_53 = Ghimj[955];
        W_54 = Ghimj[956];
        W_55 = Ghimj[957];
        W_56 = Ghimj[958];
        W_57 = Ghimj[959];
        W_58 = Ghimj[960];
        W_61 = Ghimj[961];
        W_62 = Ghimj[962];
        W_63 = Ghimj[963];
        W_64 = Ghimj[964];
        W_65 = Ghimj[965];
        W_66 = Ghimj[966];
        W_67 = Ghimj[967];
        W_68 = Ghimj[968];
        W_69 = Ghimj[969];
        W_70 = Ghimj[970];
        W_71 = Ghimj[971];
        W_72 = Ghimj[972];
        W_73 = Ghimj[973];
        W_74 = Ghimj[974];
        W_75 = Ghimj[975];
        W_76 = Ghimj[976];
        W_77 = Ghimj[977];
        W_78 = Ghimj[978];
        W_79 = Ghimj[979];
        W_81 = Ghimj[980];
        W_82 = Ghimj[981];
        W_84 = Ghimj[982];
        W_85 = Ghimj[983];
        W_86 = Ghimj[984];
        W_87 = Ghimj[985];
        W_88 = Ghimj[986];
        W_89 = Ghimj[987];
        W_91 = Ghimj[988];
        W_92 = Ghimj[989];
        W_93 = Ghimj[990];
        W_94 = Ghimj[991];
        W_95 = Ghimj[992];
        W_96 = Ghimj[993];
        W_97 = Ghimj[994];
        W_98 = Ghimj[995];
        W_99 = Ghimj[996];
        W_100 = Ghimj[997];
        W_101 = Ghimj[998];
        W_102 = Ghimj[999];
        W_103 = Ghimj[1000];
        W_104 = Ghimj[1001];
        W_105 = Ghimj[1002];
        W_106 = Ghimj[1003];
        W_107 = Ghimj[1004];
        W_108 = Ghimj[1005];
        W_109 = Ghimj[1006];
        W_110 = Ghimj[1007];
        W_111 = Ghimj[1008];
        W_112 = Ghimj[1009];
        W_113 = Ghimj[1010];
        W_114 = Ghimj[1011];
        W_115 = Ghimj[1012];
        W_116 = Ghimj[1013];
        W_117 = Ghimj[1014];
        W_118 = Ghimj[1015];
        W_119 = Ghimj[1016];
        W_120 = Ghimj[1017];
        W_121 = Ghimj[1018];
        W_122 = Ghimj[1019];
        W_123 = Ghimj[1020];
        W_124 = Ghimj[1021];
        W_125 = Ghimj[1022];
        W_126 = Ghimj[1023];
        W_127 = Ghimj[1024];
        W_128 = Ghimj[1025];
        W_129 = Ghimj[1026];
        W_130 = Ghimj[1027];
        W_131 = Ghimj[1028];
        W_132 = Ghimj[1029];
        W_133 = Ghimj[1030];
        W_134 = Ghimj[1031];
        W_135 = Ghimj[1032];
        W_136 = Ghimj[1033];
        W_137 = Ghimj[1034];
        W_138 = Ghimj[1035];
        a = - W_40/ Ghimj[260];
        W_40 = -a;
        W_126 = W_126+ a *Ghimj[261];
        a = - W_44/ Ghimj[268];
        W_44 = -a;
        W_126 = W_126+ a *Ghimj[269];
        a = - W_45/ Ghimj[270];
        W_45 = -a;
        W_126 = W_126+ a *Ghimj[271];
        a = - W_47/ Ghimj[276];
        W_47 = -a;
        W_126 = W_126+ a *Ghimj[277];
        a = - W_48/ Ghimj[278];
        W_48 = -a;
        W_126 = W_126+ a *Ghimj[279];
        a = - W_49/ Ghimj[280];
        W_49 = -a;
        W_126 = W_126+ a *Ghimj[281];
        a = - W_52/ Ghimj[288];
        W_52 = -a;
        W_126 = W_126+ a *Ghimj[289];
        a = - W_53/ Ghimj[290];
        W_53 = -a;
        W_126 = W_126+ a *Ghimj[291];
        a = - W_54/ Ghimj[292];
        W_54 = -a;
        W_126 = W_126+ a *Ghimj[293];
        a = - W_55/ Ghimj[294];
        W_55 = -a;
        W_126 = W_126+ a *Ghimj[295];
        a = - W_56/ Ghimj[296];
        W_56 = -a;
        W_65 = W_65+ a *Ghimj[297];
        W_81 = W_81+ a *Ghimj[298];
        W_126 = W_126+ a *Ghimj[299];
        a = - W_57/ Ghimj[300];
        W_57 = -a;
        W_120 = W_120+ a *Ghimj[301];
        W_126 = W_126+ a *Ghimj[302];
        a = - W_58/ Ghimj[303];
        W_58 = -a;
        W_91 = W_91+ a *Ghimj[304];
        W_126 = W_126+ a *Ghimj[305];
        a = - W_61/ Ghimj[315];
        W_61 = -a;
        W_70 = W_70+ a *Ghimj[316];
        W_87 = W_87+ a *Ghimj[317];
        W_126 = W_126+ a *Ghimj[318];
        a = - W_62/ Ghimj[319];
        W_62 = -a;
        W_93 = W_93+ a *Ghimj[320];
        W_126 = W_126+ a *Ghimj[321];
        W_133 = W_133+ a *Ghimj[322];
        a = - W_63/ Ghimj[323];
        W_63 = -a;
        W_121 = W_121+ a *Ghimj[324];
        W_126 = W_126+ a *Ghimj[325];
        W_137 = W_137+ a *Ghimj[326];
        a = - W_64/ Ghimj[327];
        W_64 = -a;
        W_113 = W_113+ a *Ghimj[328];
        W_126 = W_126+ a *Ghimj[329];
        W_135 = W_135+ a *Ghimj[330];
        a = - W_65/ Ghimj[331];
        W_65 = -a;
        W_114 = W_114+ a *Ghimj[332];
        W_126 = W_126+ a *Ghimj[333];
        W_132 = W_132+ a *Ghimj[334];
        a = - W_66/ Ghimj[335];
        W_66 = -a;
        W_109 = W_109+ a *Ghimj[336];
        W_126 = W_126+ a *Ghimj[337];
        W_137 = W_137+ a *Ghimj[338];
        a = - W_67/ Ghimj[339];
        W_67 = -a;
        W_115 = W_115+ a *Ghimj[340];
        W_126 = W_126+ a *Ghimj[341];
        W_137 = W_137+ a *Ghimj[342];
        a = - W_68/ Ghimj[343];
        W_68 = -a;
        W_99 = W_99+ a *Ghimj[344];
        W_126 = W_126+ a *Ghimj[345];
        W_137 = W_137+ a *Ghimj[346];
        a = - W_69/ Ghimj[347];
        W_69 = -a;
        W_93 = W_93+ a *Ghimj[348];
        W_126 = W_126+ a *Ghimj[349];
        W_137 = W_137+ a *Ghimj[350];
        a = - W_70/ Ghimj[352];
        W_70 = -a;
        W_84 = W_84+ a *Ghimj[353];
        W_87 = W_87+ a *Ghimj[354];
        W_126 = W_126+ a *Ghimj[355];
        a = - W_71/ Ghimj[356];
        W_71 = -a;
        W_117 = W_117+ a *Ghimj[357];
        W_126 = W_126+ a *Ghimj[358];
        W_137 = W_137+ a *Ghimj[359];
        a = - W_72/ Ghimj[360];
        W_72 = -a;
        W_94 = W_94+ a *Ghimj[361];
        W_126 = W_126+ a *Ghimj[362];
        W_137 = W_137+ a *Ghimj[363];
        a = - W_73/ Ghimj[364];
        W_73 = -a;
        W_126 = W_126+ a *Ghimj[365];
        W_135 = W_135+ a *Ghimj[366];
        W_137 = W_137+ a *Ghimj[367];
        a = - W_74/ Ghimj[368];
        W_74 = -a;
        W_117 = W_117+ a *Ghimj[369];
        W_121 = W_121+ a *Ghimj[370];
        W_125 = W_125+ a *Ghimj[371];
        W_126 = W_126+ a *Ghimj[372];
        W_137 = W_137+ a *Ghimj[373];
        a = - W_75/ Ghimj[374];
        W_75 = -a;
        W_120 = W_120+ a *Ghimj[375];
        W_126 = W_126+ a *Ghimj[376];
        a = - W_76/ Ghimj[377];
        W_76 = -a;
        W_87 = W_87+ a *Ghimj[378];
        W_126 = W_126+ a *Ghimj[379];
        W_133 = W_133+ a *Ghimj[380];
        W_135 = W_135+ a *Ghimj[381];
        a = - W_77/ Ghimj[382];
        W_77 = -a;
        W_121 = W_121+ a *Ghimj[383];
        W_126 = W_126+ a *Ghimj[384];
        W_135 = W_135+ a *Ghimj[385];
        a = - W_78/ Ghimj[386];
        W_78 = -a;
        W_103 = W_103+ a *Ghimj[387];
        W_106 = W_106+ a *Ghimj[388];
        W_107 = W_107+ a *Ghimj[389];
        W_110 = W_110+ a *Ghimj[390];
        W_124 = W_124+ a *Ghimj[391];
        W_126 = W_126+ a *Ghimj[392];
        a = - W_79/ Ghimj[393];
        W_79 = -a;
        W_102 = W_102+ a *Ghimj[394];
        W_126 = W_126+ a *Ghimj[395];
        W_137 = W_137+ a *Ghimj[396];
        a = - W_81/ Ghimj[405];
        W_81 = -a;
        W_114 = W_114+ a *Ghimj[406];
        W_124 = W_124+ a *Ghimj[407];
        W_126 = W_126+ a *Ghimj[408];
        W_127 = W_127+ a *Ghimj[409];
        W_129 = W_129+ a *Ghimj[410];
        W_136 = W_136+ a *Ghimj[411];
        a = - W_82/ Ghimj[412];
        W_82 = -a;
        W_113 = W_113+ a *Ghimj[413];
        W_126 = W_126+ a *Ghimj[414];
        W_137 = W_137+ a *Ghimj[415];
        a = - W_84/ Ghimj[421];
        W_84 = -a;
        W_92 = W_92+ a *Ghimj[422];
        W_124 = W_124+ a *Ghimj[423];
        W_135 = W_135+ a *Ghimj[424];
        W_137 = W_137+ a *Ghimj[425];
        a = - W_85/ Ghimj[427];
        W_85 = -a;
        W_102 = W_102+ a *Ghimj[428];
        W_111 = W_111+ a *Ghimj[429];
        W_125 = W_125+ a *Ghimj[430];
        W_126 = W_126+ a *Ghimj[431];
        W_133 = W_133+ a *Ghimj[432];
        W_137 = W_137+ a *Ghimj[433];
        a = - W_86/ Ghimj[436];
        W_86 = -a;
        W_93 = W_93+ a *Ghimj[437];
        W_125 = W_125+ a *Ghimj[438];
        W_126 = W_126+ a *Ghimj[439];
        W_133 = W_133+ a *Ghimj[440];
        W_137 = W_137+ a *Ghimj[441];
        a = - W_87/ Ghimj[444];
        W_87 = -a;
        W_92 = W_92+ a *Ghimj[445];
        W_124 = W_124+ a *Ghimj[446];
        W_126 = W_126+ a *Ghimj[447];
        W_135 = W_135+ a *Ghimj[448];
        W_137 = W_137+ a *Ghimj[449];
        a = - W_88/ Ghimj[450];
        W_88 = -a;
        W_103 = W_103+ a *Ghimj[451];
        W_106 = W_106+ a *Ghimj[452];
        W_124 = W_124+ a *Ghimj[453];
        W_126 = W_126+ a *Ghimj[454];
        W_127 = W_127+ a *Ghimj[455];
        W_137 = W_137+ a *Ghimj[456];
        a = - W_89/ Ghimj[457];
        W_89 = -a;
        W_93 = W_93+ a *Ghimj[458];
        W_94 = W_94+ a *Ghimj[459];
        W_102 = W_102+ a *Ghimj[460];
        W_107 = W_107+ a *Ghimj[461];
        W_109 = W_109+ a *Ghimj[462];
        W_113 = W_113+ a *Ghimj[463];
        W_117 = W_117+ a *Ghimj[464];
        W_124 = W_124+ a *Ghimj[465];
        W_125 = W_125+ a *Ghimj[466];
        W_126 = W_126+ a *Ghimj[467];
        a = - W_91/ Ghimj[481];
        W_91 = -a;
        W_106 = W_106+ a *Ghimj[482];
        W_109 = W_109+ a *Ghimj[483];
        W_126 = W_126+ a *Ghimj[484];
        W_133 = W_133+ a *Ghimj[485];
        W_136 = W_136+ a *Ghimj[486];
        a = - W_92/ Ghimj[489];
        W_92 = -a;
        W_124 = W_124+ a *Ghimj[490];
        W_126 = W_126+ a *Ghimj[491];
        W_133 = W_133+ a *Ghimj[492];
        W_135 = W_135+ a *Ghimj[493];
        W_137 = W_137+ a *Ghimj[494];
        a = - W_93/ Ghimj[497];
        W_93 = -a;
        W_125 = W_125+ a *Ghimj[498];
        W_126 = W_126+ a *Ghimj[499];
        W_133 = W_133+ a *Ghimj[500];
        W_137 = W_137+ a *Ghimj[501];
        a = - W_94/ Ghimj[505];
        W_94 = -a;
        W_125 = W_125+ a *Ghimj[506];
        W_126 = W_126+ a *Ghimj[507];
        W_133 = W_133+ a *Ghimj[508];
        W_137 = W_137+ a *Ghimj[509];
        a = - W_95/ Ghimj[514];
        W_95 = -a;
        W_96 = W_96+ a *Ghimj[515];
        W_98 = W_98+ a *Ghimj[516];
        W_103 = W_103+ a *Ghimj[517];
        W_106 = W_106+ a *Ghimj[518];
        W_107 = W_107+ a *Ghimj[519];
        W_109 = W_109+ a *Ghimj[520];
        W_110 = W_110+ a *Ghimj[521];
        W_113 = W_113+ a *Ghimj[522];
        W_119 = W_119+ a *Ghimj[523];
        W_121 = W_121+ a *Ghimj[524];
        W_124 = W_124+ a *Ghimj[525];
        W_125 = W_125+ a *Ghimj[526];
        W_126 = W_126+ a *Ghimj[527];
        W_127 = W_127+ a *Ghimj[528];
        W_129 = W_129+ a *Ghimj[529];
        W_130 = W_130+ a *Ghimj[530];
        W_133 = W_133+ a *Ghimj[531];
        W_135 = W_135+ a *Ghimj[532];
        W_136 = W_136+ a *Ghimj[533];
        W_137 = W_137+ a *Ghimj[534];
        a = - W_96/ Ghimj[538];
        W_96 = -a;
        W_107 = W_107+ a *Ghimj[539];
        W_108 = W_108+ a *Ghimj[540];
        W_109 = W_109+ a *Ghimj[541];
        W_110 = W_110+ a *Ghimj[542];
        W_113 = W_113+ a *Ghimj[543];
        W_124 = W_124+ a *Ghimj[544];
        W_125 = W_125+ a *Ghimj[545];
        W_126 = W_126+ a *Ghimj[546];
        W_133 = W_133+ a *Ghimj[547];
        W_137 = W_137+ a *Ghimj[548];
        a = - W_97/ Ghimj[549];
        W_97 = -a;
        W_98 = W_98+ a *Ghimj[550];
        W_120 = W_120+ a *Ghimj[551];
        W_122 = W_122+ a *Ghimj[552];
        W_126 = W_126+ a *Ghimj[553];
        W_127 = W_127+ a *Ghimj[554];
        W_130 = W_130+ a *Ghimj[555];
        W_137 = W_137+ a *Ghimj[556];
        a = - W_98/ Ghimj[557];
        W_98 = -a;
        W_107 = W_107+ a *Ghimj[558];
        W_120 = W_120+ a *Ghimj[559];
        W_124 = W_124+ a *Ghimj[560];
        W_126 = W_126+ a *Ghimj[561];
        W_127 = W_127+ a *Ghimj[562];
        a = - W_99/ Ghimj[565];
        W_99 = -a;
        W_102 = W_102+ a *Ghimj[566];
        W_111 = W_111+ a *Ghimj[567];
        W_125 = W_125+ a *Ghimj[568];
        W_126 = W_126+ a *Ghimj[569];
        W_133 = W_133+ a *Ghimj[570];
        W_137 = W_137+ a *Ghimj[571];
        a = - W_100/ Ghimj[573];
        W_100 = -a;
        W_105 = W_105+ a *Ghimj[574];
        W_112 = W_112+ a *Ghimj[575];
        W_116 = W_116+ a *Ghimj[576];
        W_118 = W_118+ a *Ghimj[577];
        W_123 = W_123+ a *Ghimj[578];
        W_126 = W_126+ a *Ghimj[579];
        W_127 = W_127+ a *Ghimj[580];
        W_129 = W_129+ a *Ghimj[581];
        W_132 = W_132+ a *Ghimj[582];
        W_134 = W_134+ a *Ghimj[583];
        W_138 = W_138+ a *Ghimj[584];
        a = - W_101/ Ghimj[586];
        W_101 = -a;
        W_105 = W_105+ a *Ghimj[587];
        W_114 = W_114+ a *Ghimj[588];
        W_116 = W_116+ a *Ghimj[589];
        W_119 = W_119+ a *Ghimj[590];
        W_123 = W_123+ a *Ghimj[591];
        W_126 = W_126+ a *Ghimj[592];
        W_128 = W_128+ a *Ghimj[593];
        W_130 = W_130+ a *Ghimj[594];
        W_135 = W_135+ a *Ghimj[595];
        W_136 = W_136+ a *Ghimj[596];
        W_138 = W_138+ a *Ghimj[597];
        a = - W_102/ Ghimj[600];
        W_102 = -a;
        W_125 = W_125+ a *Ghimj[601];
        W_126 = W_126+ a *Ghimj[602];
        W_133 = W_133+ a *Ghimj[603];
        W_137 = W_137+ a *Ghimj[604];
        a = - W_103/ Ghimj[605];
        W_103 = -a;
        W_124 = W_124+ a *Ghimj[606];
        W_126 = W_126+ a *Ghimj[607];
        W_127 = W_127+ a *Ghimj[608];
        W_129 = W_129+ a *Ghimj[609];
        a = - W_104/ Ghimj[610];
        W_104 = -a;
        W_125 = W_125+ a *Ghimj[611];
        W_126 = W_126+ a *Ghimj[612];
        W_127 = W_127+ a *Ghimj[613];
        W_129 = W_129+ a *Ghimj[614];
        W_137 = W_137+ a *Ghimj[615];
        a = - W_105/ Ghimj[616];
        W_105 = -a;
        W_128 = W_128+ a *Ghimj[617];
        W_129 = W_129+ a *Ghimj[618];
        W_132 = W_132+ a *Ghimj[619];
        W_135 = W_135+ a *Ghimj[620];
        W_138 = W_138+ a *Ghimj[621];
        a = - W_106/ Ghimj[622];
        W_106 = -a;
        W_124 = W_124+ a *Ghimj[623];
        W_126 = W_126+ a *Ghimj[624];
        W_136 = W_136+ a *Ghimj[625];
        a = - W_107/ Ghimj[626];
        W_107 = -a;
        W_124 = W_124+ a *Ghimj[627];
        W_126 = W_126+ a *Ghimj[628];
        W_136 = W_136+ a *Ghimj[629];
        a = - W_108/ Ghimj[636];
        W_108 = -a;
        W_109 = W_109+ a *Ghimj[637];
        W_113 = W_113+ a *Ghimj[638];
        W_115 = W_115+ a *Ghimj[639];
        W_124 = W_124+ a *Ghimj[640];
        W_125 = W_125+ a *Ghimj[641];
        W_126 = W_126+ a *Ghimj[642];
        W_133 = W_133+ a *Ghimj[643];
        W_135 = W_135+ a *Ghimj[644];
        W_136 = W_136+ a *Ghimj[645];
        W_137 = W_137+ a *Ghimj[646];
        a = - W_109/ Ghimj[648];
        W_109 = -a;
        W_124 = W_124+ a *Ghimj[649];
        W_125 = W_125+ a *Ghimj[650];
        W_126 = W_126+ a *Ghimj[651];
        W_133 = W_133+ a *Ghimj[652];
        W_136 = W_136+ a *Ghimj[653];
        W_137 = W_137+ a *Ghimj[654];
        a = - W_110/ Ghimj[659];
        W_110 = -a;
        W_124 = W_124+ a *Ghimj[660];
        W_125 = W_125+ a *Ghimj[661];
        W_126 = W_126+ a *Ghimj[662];
        W_133 = W_133+ a *Ghimj[663];
        W_136 = W_136+ a *Ghimj[664];
        W_137 = W_137+ a *Ghimj[665];
        a = - W_111/ Ghimj[669];
        W_111 = -a;
        W_115 = W_115+ a *Ghimj[670];
        W_124 = W_124+ a *Ghimj[671];
        W_125 = W_125+ a *Ghimj[672];
        W_126 = W_126+ a *Ghimj[673];
        W_133 = W_133+ a *Ghimj[674];
        W_136 = W_136+ a *Ghimj[675];
        W_137 = W_137+ a *Ghimj[676];
        a = - W_112/ Ghimj[677];
        W_112 = -a;
        W_116 = W_116+ a *Ghimj[678];
        W_123 = W_123+ a *Ghimj[679];
        W_126 = W_126+ a *Ghimj[680];
        W_128 = W_128+ a *Ghimj[681];
        W_134 = W_134+ a *Ghimj[682];
        W_137 = W_137+ a *Ghimj[683];
        W_138 = W_138+ a *Ghimj[684];
        a = - W_113/ Ghimj[689];
        W_113 = -a;
        W_124 = W_124+ a *Ghimj[690];
        W_125 = W_125+ a *Ghimj[691];
        W_126 = W_126+ a *Ghimj[692];
        W_133 = W_133+ a *Ghimj[693];
        W_135 = W_135+ a *Ghimj[694];
        W_136 = W_136+ a *Ghimj[695];
        W_137 = W_137+ a *Ghimj[696];
        a = - W_114/ Ghimj[697];
        W_114 = -a;
        W_126 = W_126+ a *Ghimj[698];
        W_127 = W_127+ a *Ghimj[699];
        W_129 = W_129+ a *Ghimj[700];
        W_132 = W_132+ a *Ghimj[701];
        W_136 = W_136+ a *Ghimj[702];
        a = - W_115/ Ghimj[706];
        W_115 = -a;
        W_124 = W_124+ a *Ghimj[707];
        W_126 = W_126+ a *Ghimj[708];
        W_127 = W_127+ a *Ghimj[709];
        W_129 = W_129+ a *Ghimj[710];
        W_133 = W_133+ a *Ghimj[711];
        W_136 = W_136+ a *Ghimj[712];
        W_137 = W_137+ a *Ghimj[713];
        a = - W_116/ Ghimj[714];
        W_116 = -a;
        W_123 = W_123+ a *Ghimj[715];
        W_127 = W_127+ a *Ghimj[716];
        W_128 = W_128+ a *Ghimj[717];
        W_131 = W_131+ a *Ghimj[718];
        W_134 = W_134+ a *Ghimj[719];
        W_135 = W_135+ a *Ghimj[720];
        W_138 = W_138+ a *Ghimj[721];
        a = - W_117/ Ghimj[731];
        W_117 = -a;
        W_121 = W_121+ a *Ghimj[732];
        W_124 = W_124+ a *Ghimj[733];
        W_125 = W_125+ a *Ghimj[734];
        W_126 = W_126+ a *Ghimj[735];
        W_127 = W_127+ a *Ghimj[736];
        W_129 = W_129+ a *Ghimj[737];
        W_133 = W_133+ a *Ghimj[738];
        W_136 = W_136+ a *Ghimj[739];
        W_137 = W_137+ a *Ghimj[740];
        a = - W_118/ Ghimj[745];
        W_118 = -a;
        W_123 = W_123+ a *Ghimj[746];
        W_125 = W_125+ a *Ghimj[747];
        W_126 = W_126+ a *Ghimj[748];
        W_127 = W_127+ a *Ghimj[749];
        W_128 = W_128+ a *Ghimj[750];
        W_129 = W_129+ a *Ghimj[751];
        W_131 = W_131+ a *Ghimj[752];
        W_132 = W_132+ a *Ghimj[753];
        W_134 = W_134+ a *Ghimj[754];
        W_135 = W_135+ a *Ghimj[755];
        W_137 = W_137+ a *Ghimj[756];
        W_138 = W_138+ a *Ghimj[757];
        a = - W_119/ Ghimj[767];
        W_119 = -a;
        W_121 = W_121+ a *Ghimj[768];
        W_124 = W_124+ a *Ghimj[769];
        W_125 = W_125+ a *Ghimj[770];
        W_126 = W_126+ a *Ghimj[771];
        W_127 = W_127+ a *Ghimj[772];
        W_129 = W_129+ a *Ghimj[773];
        W_133 = W_133+ a *Ghimj[774];
        W_136 = W_136+ a *Ghimj[775];
        W_137 = W_137+ a *Ghimj[776];
        a = - W_120/ Ghimj[787];
        W_120 = -a;
        W_122 = W_122+ a *Ghimj[788];
        W_124 = W_124+ a *Ghimj[789];
        W_126 = W_126+ a *Ghimj[790];
        W_127 = W_127+ a *Ghimj[791];
        W_128 = W_128+ a *Ghimj[792];
        W_130 = W_130+ a *Ghimj[793];
        W_133 = W_133+ a *Ghimj[794];
        W_135 = W_135+ a *Ghimj[795];
        W_136 = W_136+ a *Ghimj[796];
        W_137 = W_137+ a *Ghimj[797];
        a = - W_121/ Ghimj[821];
        W_121 = -a;
        W_124 = W_124+ a *Ghimj[822];
        W_125 = W_125+ a *Ghimj[823];
        W_126 = W_126+ a *Ghimj[824];
        W_127 = W_127+ a *Ghimj[825];
        W_129 = W_129+ a *Ghimj[826];
        W_133 = W_133+ a *Ghimj[827];
        W_135 = W_135+ a *Ghimj[828];
        W_136 = W_136+ a *Ghimj[829];
        W_137 = W_137+ a *Ghimj[830];
        a = - W_122/ Ghimj[847];
        W_122 = -a;
        W_124 = W_124+ a *Ghimj[848];
        W_125 = W_125+ a *Ghimj[849];
        W_126 = W_126+ a *Ghimj[850];
        W_127 = W_127+ a *Ghimj[851];
        W_128 = W_128+ a *Ghimj[852];
        W_129 = W_129+ a *Ghimj[853];
        W_130 = W_130+ a *Ghimj[854];
        W_131 = W_131+ a *Ghimj[855];
        W_133 = W_133+ a *Ghimj[856];
        W_135 = W_135+ a *Ghimj[857];
        W_136 = W_136+ a *Ghimj[858];
        W_137 = W_137+ a *Ghimj[859];
        W_138 = W_138+ a *Ghimj[860];
        a = - W_123/ Ghimj[869];
        W_123 = -a;
        W_124 = W_124+ a *Ghimj[870];
        W_125 = W_125+ a *Ghimj[871];
        W_126 = W_126+ a *Ghimj[872];
        W_127 = W_127+ a *Ghimj[873];
        W_128 = W_128+ a *Ghimj[874];
        W_129 = W_129+ a *Ghimj[875];
        W_130 = W_130+ a *Ghimj[876];
        W_131 = W_131+ a *Ghimj[877];
        W_132 = W_132+ a *Ghimj[878];
        W_133 = W_133+ a *Ghimj[879];
        W_134 = W_134+ a *Ghimj[880];
        W_135 = W_135+ a *Ghimj[881];
        W_136 = W_136+ a *Ghimj[882];
        W_137 = W_137+ a *Ghimj[883];
        W_138 = W_138+ a *Ghimj[884];
        a = - W_124/ Ghimj[896];
        W_124 = -a;
        W_125 = W_125+ a *Ghimj[897];
        W_126 = W_126+ a *Ghimj[898];
        W_127 = W_127+ a *Ghimj[899];
        W_128 = W_128+ a *Ghimj[900];
        W_129 = W_129+ a *Ghimj[901];
        W_130 = W_130+ a *Ghimj[902];
        W_131 = W_131+ a *Ghimj[903];
        W_132 = W_132+ a *Ghimj[904];
        W_133 = W_133+ a *Ghimj[905];
        W_135 = W_135+ a *Ghimj[906];
        W_136 = W_136+ a *Ghimj[907];
        W_137 = W_137+ a *Ghimj[908];
        W_138 = W_138+ a *Ghimj[909];
        a = - W_125/ Ghimj[934];
        W_125 = -a;
        W_126 = W_126+ a *Ghimj[935];
        W_127 = W_127+ a *Ghimj[936];
        W_128 = W_128+ a *Ghimj[937];
        W_129 = W_129+ a *Ghimj[938];
        W_130 = W_130+ a *Ghimj[939];
        W_131 = W_131+ a *Ghimj[940];
        W_132 = W_132+ a *Ghimj[941];
        W_133 = W_133+ a *Ghimj[942];
        W_134 = W_134+ a *Ghimj[943];
        W_135 = W_135+ a *Ghimj[944];
        W_136 = W_136+ a *Ghimj[945];
        W_137 = W_137+ a *Ghimj[946];
        W_138 = W_138+ a *Ghimj[947];
        Ghimj[948] = W_40;
        Ghimj[949] = W_44;
        Ghimj[950] = W_45;
        Ghimj[951] = W_47;
        Ghimj[952] = W_48;
        Ghimj[953] = W_49;
        Ghimj[954] = W_52;
        Ghimj[955] = W_53;
        Ghimj[956] = W_54;
        Ghimj[957] = W_55;
        Ghimj[958] = W_56;
        Ghimj[959] = W_57;
        Ghimj[960] = W_58;
        Ghimj[961] = W_61;
        Ghimj[962] = W_62;
        Ghimj[963] = W_63;
        Ghimj[964] = W_64;
        Ghimj[965] = W_65;
        Ghimj[966] = W_66;
        Ghimj[967] = W_67;
        Ghimj[968] = W_68;
        Ghimj[969] = W_69;
        Ghimj[970] = W_70;
        Ghimj[971] = W_71;
        Ghimj[972] = W_72;
        Ghimj[973] = W_73;
        Ghimj[974] = W_74;
        Ghimj[975] = W_75;
        Ghimj[976] = W_76;
        Ghimj[977] = W_77;
        Ghimj[978] = W_78;
        Ghimj[979] = W_79;
        Ghimj[980] = W_81;
        Ghimj[981] = W_82;
        Ghimj[982] = W_84;
        Ghimj[983] = W_85;
        Ghimj[984] = W_86;
        Ghimj[985] = W_87;
        Ghimj[986] = W_88;
        Ghimj[987] = W_89;
        Ghimj[988] = W_91;
        Ghimj[989] = W_92;
        Ghimj[990] = W_93;
        Ghimj[991] = W_94;
        Ghimj[992] = W_95;
        Ghimj[993] = W_96;
        Ghimj[994] = W_97;
        Ghimj[995] = W_98;
        Ghimj[996] = W_99;
        Ghimj[997] = W_100;
        Ghimj[998] = W_101;
        Ghimj[999] = W_102;
        Ghimj[1000] = W_103;
        Ghimj[1001] = W_104;
        Ghimj[1002] = W_105;
        Ghimj[1003] = W_106;
        Ghimj[1004] = W_107;
        Ghimj[1005] = W_108;
        Ghimj[1006] = W_109;
        Ghimj[1007] = W_110;
        Ghimj[1008] = W_111;
        Ghimj[1009] = W_112;
        Ghimj[1010] = W_113;
        Ghimj[1011] = W_114;
        Ghimj[1012] = W_115;
        Ghimj[1013] = W_116;
        Ghimj[1014] = W_117;
        Ghimj[1015] = W_118;
        Ghimj[1016] = W_119;
        Ghimj[1017] = W_120;
        Ghimj[1018] = W_121;
        Ghimj[1019] = W_122;
        Ghimj[1020] = W_123;
        Ghimj[1021] = W_124;
        Ghimj[1022] = W_125;
        Ghimj[1023] = W_126;
        Ghimj[1024] = W_127;
        Ghimj[1025] = W_128;
        Ghimj[1026] = W_129;
        Ghimj[1027] = W_130;
        Ghimj[1028] = W_131;
        Ghimj[1029] = W_132;
        Ghimj[1030] = W_133;
        Ghimj[1031] = W_134;
        Ghimj[1032] = W_135;
        Ghimj[1033] = W_136;
        Ghimj[1034] = W_137;
        Ghimj[1035] = W_138;
        W_1 = Ghimj[1036];
        W_39 = Ghimj[1037];
        W_41 = Ghimj[1038];
        W_42 = Ghimj[1039];
        W_43 = Ghimj[1040];
        W_50 = Ghimj[1041];
        W_52 = Ghimj[1042];
        W_54 = Ghimj[1043];
        W_55 = Ghimj[1044];
        W_57 = Ghimj[1045];
        W_75 = Ghimj[1046];
        W_80 = Ghimj[1047];
        W_83 = Ghimj[1048];
        W_88 = Ghimj[1049];
        W_90 = Ghimj[1050];
        W_97 = Ghimj[1051];
        W_98 = Ghimj[1052];
        W_100 = Ghimj[1053];
        W_103 = Ghimj[1054];
        W_104 = Ghimj[1055];
        W_105 = Ghimj[1056];
        W_106 = Ghimj[1057];
        W_107 = Ghimj[1058];
        W_112 = Ghimj[1059];
        W_114 = Ghimj[1060];
        W_116 = Ghimj[1061];
        W_118 = Ghimj[1062];
        W_119 = Ghimj[1063];
        W_120 = Ghimj[1064];
        W_121 = Ghimj[1065];
        W_122 = Ghimj[1066];
        W_123 = Ghimj[1067];
        W_124 = Ghimj[1068];
        W_125 = Ghimj[1069];
        W_126 = Ghimj[1070];
        W_127 = Ghimj[1071];
        W_128 = Ghimj[1072];
        W_129 = Ghimj[1073];
        W_130 = Ghimj[1074];
        W_131 = Ghimj[1075];
        W_132 = Ghimj[1076];
        W_133 = Ghimj[1077];
        W_134 = Ghimj[1078];
        W_135 = Ghimj[1079];
        W_136 = Ghimj[1080];
        W_137 = Ghimj[1081];
        W_138 = Ghimj[1082];
        a = - W_1/ Ghimj[1];
        W_1 = -a;
        a = - W_39/ Ghimj[258];
        W_39 = -a;
        W_134 = W_134+ a *Ghimj[259];
        a = - W_41/ Ghimj[262];
        W_41 = -a;
        W_120 = W_120+ a *Ghimj[263];
        a = - W_42/ Ghimj[264];
        W_42 = -a;
        W_120 = W_120+ a *Ghimj[265];
        a = - W_43/ Ghimj[266];
        W_43 = -a;
        W_120 = W_120+ a *Ghimj[267];
        a = - W_50/ Ghimj[282];
        W_50 = -a;
        W_83 = W_83+ a *Ghimj[283];
        W_138 = W_138+ a *Ghimj[284];
        a = - W_52/ Ghimj[288];
        W_52 = -a;
        W_126 = W_126+ a *Ghimj[289];
        a = - W_54/ Ghimj[292];
        W_54 = -a;
        W_126 = W_126+ a *Ghimj[293];
        a = - W_55/ Ghimj[294];
        W_55 = -a;
        W_126 = W_126+ a *Ghimj[295];
        a = - W_57/ Ghimj[300];
        W_57 = -a;
        W_120 = W_120+ a *Ghimj[301];
        W_126 = W_126+ a *Ghimj[302];
        a = - W_75/ Ghimj[374];
        W_75 = -a;
        W_120 = W_120+ a *Ghimj[375];
        W_126 = W_126+ a *Ghimj[376];
        a = - W_80/ Ghimj[397];
        W_80 = -a;
        W_90 = W_90+ a *Ghimj[398];
        W_112 = W_112+ a *Ghimj[399];
        W_116 = W_116+ a *Ghimj[400];
        W_127 = W_127+ a *Ghimj[401];
        W_129 = W_129+ a *Ghimj[402];
        W_134 = W_134+ a *Ghimj[403];
        W_138 = W_138+ a *Ghimj[404];
        a = - W_83/ Ghimj[416];
        W_83 = -a;
        W_128 = W_128+ a *Ghimj[417];
        W_135 = W_135+ a *Ghimj[418];
        W_136 = W_136+ a *Ghimj[419];
        W_138 = W_138+ a *Ghimj[420];
        a = - W_88/ Ghimj[450];
        W_88 = -a;
        W_103 = W_103+ a *Ghimj[451];
        W_106 = W_106+ a *Ghimj[452];
        W_124 = W_124+ a *Ghimj[453];
        W_126 = W_126+ a *Ghimj[454];
        W_127 = W_127+ a *Ghimj[455];
        W_137 = W_137+ a *Ghimj[456];
        a = - W_90/ Ghimj[469];
        W_90 = -a;
        W_100 = W_100+ a *Ghimj[470];
        W_105 = W_105+ a *Ghimj[471];
        W_112 = W_112+ a *Ghimj[472];
        W_116 = W_116+ a *Ghimj[473];
        W_118 = W_118+ a *Ghimj[474];
        W_123 = W_123+ a *Ghimj[475];
        W_127 = W_127+ a *Ghimj[476];
        W_129 = W_129+ a *Ghimj[477];
        W_132 = W_132+ a *Ghimj[478];
        W_134 = W_134+ a *Ghimj[479];
        W_138 = W_138+ a *Ghimj[480];
        a = - W_97/ Ghimj[549];
        W_97 = -a;
        W_98 = W_98+ a *Ghimj[550];
        W_120 = W_120+ a *Ghimj[551];
        W_122 = W_122+ a *Ghimj[552];
        W_126 = W_126+ a *Ghimj[553];
        W_127 = W_127+ a *Ghimj[554];
        W_130 = W_130+ a *Ghimj[555];
        W_137 = W_137+ a *Ghimj[556];
        a = - W_98/ Ghimj[557];
        W_98 = -a;
        W_107 = W_107+ a *Ghimj[558];
        W_120 = W_120+ a *Ghimj[559];
        W_124 = W_124+ a *Ghimj[560];
        W_126 = W_126+ a *Ghimj[561];
        W_127 = W_127+ a *Ghimj[562];
        a = - W_100/ Ghimj[573];
        W_100 = -a;
        W_105 = W_105+ a *Ghimj[574];
        W_112 = W_112+ a *Ghimj[575];
        W_116 = W_116+ a *Ghimj[576];
        W_118 = W_118+ a *Ghimj[577];
        W_123 = W_123+ a *Ghimj[578];
        W_126 = W_126+ a *Ghimj[579];
        W_127 = W_127+ a *Ghimj[580];
        W_129 = W_129+ a *Ghimj[581];
        W_132 = W_132+ a *Ghimj[582];
        W_134 = W_134+ a *Ghimj[583];
        W_138 = W_138+ a *Ghimj[584];
        a = - W_103/ Ghimj[605];
        W_103 = -a;
        W_124 = W_124+ a *Ghimj[606];
        W_126 = W_126+ a *Ghimj[607];
        W_127 = W_127+ a *Ghimj[608];
        W_129 = W_129+ a *Ghimj[609];
        a = - W_104/ Ghimj[610];
        W_104 = -a;
        W_125 = W_125+ a *Ghimj[611];
        W_126 = W_126+ a *Ghimj[612];
        W_127 = W_127+ a *Ghimj[613];
        W_129 = W_129+ a *Ghimj[614];
        W_137 = W_137+ a *Ghimj[615];
        a = - W_105/ Ghimj[616];
        W_105 = -a;
        W_128 = W_128+ a *Ghimj[617];
        W_129 = W_129+ a *Ghimj[618];
        W_132 = W_132+ a *Ghimj[619];
        W_135 = W_135+ a *Ghimj[620];
        W_138 = W_138+ a *Ghimj[621];
        a = - W_106/ Ghimj[622];
        W_106 = -a;
        W_124 = W_124+ a *Ghimj[623];
        W_126 = W_126+ a *Ghimj[624];
        W_136 = W_136+ a *Ghimj[625];
        a = - W_107/ Ghimj[626];
        W_107 = -a;
        W_124 = W_124+ a *Ghimj[627];
        W_126 = W_126+ a *Ghimj[628];
        W_136 = W_136+ a *Ghimj[629];
        a = - W_112/ Ghimj[677];
        W_112 = -a;
        W_116 = W_116+ a *Ghimj[678];
        W_123 = W_123+ a *Ghimj[679];
        W_126 = W_126+ a *Ghimj[680];
        W_128 = W_128+ a *Ghimj[681];
        W_134 = W_134+ a *Ghimj[682];
        W_137 = W_137+ a *Ghimj[683];
        W_138 = W_138+ a *Ghimj[684];
        a = - W_114/ Ghimj[697];
        W_114 = -a;
        W_126 = W_126+ a *Ghimj[698];
        W_127 = W_127+ a *Ghimj[699];
        W_129 = W_129+ a *Ghimj[700];
        W_132 = W_132+ a *Ghimj[701];
        W_136 = W_136+ a *Ghimj[702];
        a = - W_116/ Ghimj[714];
        W_116 = -a;
        W_123 = W_123+ a *Ghimj[715];
        W_127 = W_127+ a *Ghimj[716];
        W_128 = W_128+ a *Ghimj[717];
        W_131 = W_131+ a *Ghimj[718];
        W_134 = W_134+ a *Ghimj[719];
        W_135 = W_135+ a *Ghimj[720];
        W_138 = W_138+ a *Ghimj[721];
        a = - W_118/ Ghimj[745];
        W_118 = -a;
        W_123 = W_123+ a *Ghimj[746];
        W_125 = W_125+ a *Ghimj[747];
        W_126 = W_126+ a *Ghimj[748];
        W_127 = W_127+ a *Ghimj[749];
        W_128 = W_128+ a *Ghimj[750];
        W_129 = W_129+ a *Ghimj[751];
        W_131 = W_131+ a *Ghimj[752];
        W_132 = W_132+ a *Ghimj[753];
        W_134 = W_134+ a *Ghimj[754];
        W_135 = W_135+ a *Ghimj[755];
        W_137 = W_137+ a *Ghimj[756];
        W_138 = W_138+ a *Ghimj[757];
        a = - W_119/ Ghimj[767];
        W_119 = -a;
        W_121 = W_121+ a *Ghimj[768];
        W_124 = W_124+ a *Ghimj[769];
        W_125 = W_125+ a *Ghimj[770];
        W_126 = W_126+ a *Ghimj[771];
        W_127 = W_127+ a *Ghimj[772];
        W_129 = W_129+ a *Ghimj[773];
        W_133 = W_133+ a *Ghimj[774];
        W_136 = W_136+ a *Ghimj[775];
        W_137 = W_137+ a *Ghimj[776];
        a = - W_120/ Ghimj[787];
        W_120 = -a;
        W_122 = W_122+ a *Ghimj[788];
        W_124 = W_124+ a *Ghimj[789];
        W_126 = W_126+ a *Ghimj[790];
        W_127 = W_127+ a *Ghimj[791];
        W_128 = W_128+ a *Ghimj[792];
        W_130 = W_130+ a *Ghimj[793];
        W_133 = W_133+ a *Ghimj[794];
        W_135 = W_135+ a *Ghimj[795];
        W_136 = W_136+ a *Ghimj[796];
        W_137 = W_137+ a *Ghimj[797];
        a = - W_121/ Ghimj[821];
        W_121 = -a;
        W_124 = W_124+ a *Ghimj[822];
        W_125 = W_125+ a *Ghimj[823];
        W_126 = W_126+ a *Ghimj[824];
        W_127 = W_127+ a *Ghimj[825];
        W_129 = W_129+ a *Ghimj[826];
        W_133 = W_133+ a *Ghimj[827];
        W_135 = W_135+ a *Ghimj[828];
        W_136 = W_136+ a *Ghimj[829];
        W_137 = W_137+ a *Ghimj[830];
        a = - W_122/ Ghimj[847];
        W_122 = -a;
        W_124 = W_124+ a *Ghimj[848];
        W_125 = W_125+ a *Ghimj[849];
        W_126 = W_126+ a *Ghimj[850];
        W_127 = W_127+ a *Ghimj[851];
        W_128 = W_128+ a *Ghimj[852];
        W_129 = W_129+ a *Ghimj[853];
        W_130 = W_130+ a *Ghimj[854];
        W_131 = W_131+ a *Ghimj[855];
        W_133 = W_133+ a *Ghimj[856];
        W_135 = W_135+ a *Ghimj[857];
        W_136 = W_136+ a *Ghimj[858];
        W_137 = W_137+ a *Ghimj[859];
        W_138 = W_138+ a *Ghimj[860];
        a = - W_123/ Ghimj[869];
        W_123 = -a;
        W_124 = W_124+ a *Ghimj[870];
        W_125 = W_125+ a *Ghimj[871];
        W_126 = W_126+ a *Ghimj[872];
        W_127 = W_127+ a *Ghimj[873];
        W_128 = W_128+ a *Ghimj[874];
        W_129 = W_129+ a *Ghimj[875];
        W_130 = W_130+ a *Ghimj[876];
        W_131 = W_131+ a *Ghimj[877];
        W_132 = W_132+ a *Ghimj[878];
        W_133 = W_133+ a *Ghimj[879];
        W_134 = W_134+ a *Ghimj[880];
        W_135 = W_135+ a *Ghimj[881];
        W_136 = W_136+ a *Ghimj[882];
        W_137 = W_137+ a *Ghimj[883];
        W_138 = W_138+ a *Ghimj[884];
        a = - W_124/ Ghimj[896];
        W_124 = -a;
        W_125 = W_125+ a *Ghimj[897];
        W_126 = W_126+ a *Ghimj[898];
        W_127 = W_127+ a *Ghimj[899];
        W_128 = W_128+ a *Ghimj[900];
        W_129 = W_129+ a *Ghimj[901];
        W_130 = W_130+ a *Ghimj[902];
        W_131 = W_131+ a *Ghimj[903];
        W_132 = W_132+ a *Ghimj[904];
        W_133 = W_133+ a *Ghimj[905];
        W_135 = W_135+ a *Ghimj[906];
        W_136 = W_136+ a *Ghimj[907];
        W_137 = W_137+ a *Ghimj[908];
        W_138 = W_138+ a *Ghimj[909];
        a = - W_125/ Ghimj[934];
        W_125 = -a;
        W_126 = W_126+ a *Ghimj[935];
        W_127 = W_127+ a *Ghimj[936];
        W_128 = W_128+ a *Ghimj[937];
        W_129 = W_129+ a *Ghimj[938];
        W_130 = W_130+ a *Ghimj[939];
        W_131 = W_131+ a *Ghimj[940];
        W_132 = W_132+ a *Ghimj[941];
        W_133 = W_133+ a *Ghimj[942];
        W_134 = W_134+ a *Ghimj[943];
        W_135 = W_135+ a *Ghimj[944];
        W_136 = W_136+ a *Ghimj[945];
        W_137 = W_137+ a *Ghimj[946];
        W_138 = W_138+ a *Ghimj[947];
        a = - W_126/ Ghimj[1023];
        W_126 = -a;
        W_127 = W_127+ a *Ghimj[1024];
        W_128 = W_128+ a *Ghimj[1025];
        W_129 = W_129+ a *Ghimj[1026];
        W_130 = W_130+ a *Ghimj[1027];
        W_131 = W_131+ a *Ghimj[1028];
        W_132 = W_132+ a *Ghimj[1029];
        W_133 = W_133+ a *Ghimj[1030];
        W_134 = W_134+ a *Ghimj[1031];
        W_135 = W_135+ a *Ghimj[1032];
        W_136 = W_136+ a *Ghimj[1033];
        W_137 = W_137+ a *Ghimj[1034];
        W_138 = W_138+ a *Ghimj[1035];
        Ghimj[1036] = W_1;
        Ghimj[1037] = W_39;
        Ghimj[1038] = W_41;
        Ghimj[1039] = W_42;
        Ghimj[1040] = W_43;
        Ghimj[1041] = W_50;
        Ghimj[1042] = W_52;
        Ghimj[1043] = W_54;
        Ghimj[1044] = W_55;
        Ghimj[1045] = W_57;
        Ghimj[1046] = W_75;
        Ghimj[1047] = W_80;
        Ghimj[1048] = W_83;
        Ghimj[1049] = W_88;
        Ghimj[1050] = W_90;
        Ghimj[1051] = W_97;
        Ghimj[1052] = W_98;
        Ghimj[1053] = W_100;
        Ghimj[1054] = W_103;
        Ghimj[1055] = W_104;
        Ghimj[1056] = W_105;
        Ghimj[1057] = W_106;
        Ghimj[1058] = W_107;
        Ghimj[1059] = W_112;
        Ghimj[1060] = W_114;
        Ghimj[1061] = W_116;
        Ghimj[1062] = W_118;
        Ghimj[1063] = W_119;
        Ghimj[1064] = W_120;
        Ghimj[1065] = W_121;
        Ghimj[1066] = W_122;
        Ghimj[1067] = W_123;
        Ghimj[1068] = W_124;
        Ghimj[1069] = W_125;
        Ghimj[1070] = W_126;
        Ghimj[1071] = W_127;
        Ghimj[1072] = W_128;
        Ghimj[1073] = W_129;
        Ghimj[1074] = W_130;
        Ghimj[1075] = W_131;
        Ghimj[1076] = W_132;
        Ghimj[1077] = W_133;
        Ghimj[1078] = W_134;
        Ghimj[1079] = W_135;
        Ghimj[1080] = W_136;
        Ghimj[1081] = W_137;
        Ghimj[1082] = W_138;
        W_40 = Ghimj[1083];
        W_44 = Ghimj[1084];
        W_45 = Ghimj[1085];
        W_47 = Ghimj[1086];
        W_48 = Ghimj[1087];
        W_49 = Ghimj[1088];
        W_52 = Ghimj[1089];
        W_53 = Ghimj[1090];
        W_54 = Ghimj[1091];
        W_55 = Ghimj[1092];
        W_57 = Ghimj[1093];
        W_61 = Ghimj[1094];
        W_63 = Ghimj[1095];
        W_67 = Ghimj[1096];
        W_70 = Ghimj[1097];
        W_73 = Ghimj[1098];
        W_74 = Ghimj[1099];
        W_75 = Ghimj[1100];
        W_76 = Ghimj[1101];
        W_77 = Ghimj[1102];
        W_78 = Ghimj[1103];
        W_79 = Ghimj[1104];
        W_83 = Ghimj[1105];
        W_84 = Ghimj[1106];
        W_86 = Ghimj[1107];
        W_87 = Ghimj[1108];
        W_88 = Ghimj[1109];
        W_92 = Ghimj[1110];
        W_93 = Ghimj[1111];
        W_97 = Ghimj[1112];
        W_98 = Ghimj[1113];
        W_101 = Ghimj[1114];
        W_102 = Ghimj[1115];
        W_103 = Ghimj[1116];
        W_104 = Ghimj[1117];
        W_105 = Ghimj[1118];
        W_106 = Ghimj[1119];
        W_107 = Ghimj[1120];
        W_110 = Ghimj[1121];
        W_111 = Ghimj[1122];
        W_112 = Ghimj[1123];
        W_114 = Ghimj[1124];
        W_115 = Ghimj[1125];
        W_116 = Ghimj[1126];
        W_117 = Ghimj[1127];
        W_118 = Ghimj[1128];
        W_119 = Ghimj[1129];
        W_120 = Ghimj[1130];
        W_121 = Ghimj[1131];
        W_122 = Ghimj[1132];
        W_123 = Ghimj[1133];
        W_124 = Ghimj[1134];
        W_125 = Ghimj[1135];
        W_126 = Ghimj[1136];
        W_127 = Ghimj[1137];
        W_128 = Ghimj[1138];
        W_129 = Ghimj[1139];
        W_130 = Ghimj[1140];
        W_131 = Ghimj[1141];
        W_132 = Ghimj[1142];
        W_133 = Ghimj[1143];
        W_134 = Ghimj[1144];
        W_135 = Ghimj[1145];
        W_136 = Ghimj[1146];
        W_137 = Ghimj[1147];
        W_138 = Ghimj[1148];
        a = - W_40/ Ghimj[260];
        W_40 = -a;
        W_126 = W_126+ a *Ghimj[261];
        a = - W_44/ Ghimj[268];
        W_44 = -a;
        W_126 = W_126+ a *Ghimj[269];
        a = - W_45/ Ghimj[270];
        W_45 = -a;
        W_126 = W_126+ a *Ghimj[271];
        a = - W_47/ Ghimj[276];
        W_47 = -a;
        W_126 = W_126+ a *Ghimj[277];
        a = - W_48/ Ghimj[278];
        W_48 = -a;
        W_126 = W_126+ a *Ghimj[279];
        a = - W_49/ Ghimj[280];
        W_49 = -a;
        W_126 = W_126+ a *Ghimj[281];
        a = - W_52/ Ghimj[288];
        W_52 = -a;
        W_126 = W_126+ a *Ghimj[289];
        a = - W_53/ Ghimj[290];
        W_53 = -a;
        W_126 = W_126+ a *Ghimj[291];
        a = - W_54/ Ghimj[292];
        W_54 = -a;
        W_126 = W_126+ a *Ghimj[293];
        a = - W_55/ Ghimj[294];
        W_55 = -a;
        W_126 = W_126+ a *Ghimj[295];
        a = - W_57/ Ghimj[300];
        W_57 = -a;
        W_120 = W_120+ a *Ghimj[301];
        W_126 = W_126+ a *Ghimj[302];
        a = - W_61/ Ghimj[315];
        W_61 = -a;
        W_70 = W_70+ a *Ghimj[316];
        W_87 = W_87+ a *Ghimj[317];
        W_126 = W_126+ a *Ghimj[318];
        a = - W_63/ Ghimj[323];
        W_63 = -a;
        W_121 = W_121+ a *Ghimj[324];
        W_126 = W_126+ a *Ghimj[325];
        W_137 = W_137+ a *Ghimj[326];
        a = - W_67/ Ghimj[339];
        W_67 = -a;
        W_115 = W_115+ a *Ghimj[340];
        W_126 = W_126+ a *Ghimj[341];
        W_137 = W_137+ a *Ghimj[342];
        a = - W_70/ Ghimj[352];
        W_70 = -a;
        W_84 = W_84+ a *Ghimj[353];
        W_87 = W_87+ a *Ghimj[354];
        W_126 = W_126+ a *Ghimj[355];
        a = - W_73/ Ghimj[364];
        W_73 = -a;
        W_126 = W_126+ a *Ghimj[365];
        W_135 = W_135+ a *Ghimj[366];
        W_137 = W_137+ a *Ghimj[367];
        a = - W_74/ Ghimj[368];
        W_74 = -a;
        W_117 = W_117+ a *Ghimj[369];
        W_121 = W_121+ a *Ghimj[370];
        W_125 = W_125+ a *Ghimj[371];
        W_126 = W_126+ a *Ghimj[372];
        W_137 = W_137+ a *Ghimj[373];
        a = - W_75/ Ghimj[374];
        W_75 = -a;
        W_120 = W_120+ a *Ghimj[375];
        W_126 = W_126+ a *Ghimj[376];
        a = - W_76/ Ghimj[377];
        W_76 = -a;
        W_87 = W_87+ a *Ghimj[378];
        W_126 = W_126+ a *Ghimj[379];
        W_133 = W_133+ a *Ghimj[380];
        W_135 = W_135+ a *Ghimj[381];
        a = - W_77/ Ghimj[382];
        W_77 = -a;
        W_121 = W_121+ a *Ghimj[383];
        W_126 = W_126+ a *Ghimj[384];
        W_135 = W_135+ a *Ghimj[385];
        a = - W_78/ Ghimj[386];
        W_78 = -a;
        W_103 = W_103+ a *Ghimj[387];
        W_106 = W_106+ a *Ghimj[388];
        W_107 = W_107+ a *Ghimj[389];
        W_110 = W_110+ a *Ghimj[390];
        W_124 = W_124+ a *Ghimj[391];
        W_126 = W_126+ a *Ghimj[392];
        a = - W_79/ Ghimj[393];
        W_79 = -a;
        W_102 = W_102+ a *Ghimj[394];
        W_126 = W_126+ a *Ghimj[395];
        W_137 = W_137+ a *Ghimj[396];
        a = - W_83/ Ghimj[416];
        W_83 = -a;
        W_128 = W_128+ a *Ghimj[417];
        W_135 = W_135+ a *Ghimj[418];
        W_136 = W_136+ a *Ghimj[419];
        W_138 = W_138+ a *Ghimj[420];
        a = - W_84/ Ghimj[421];
        W_84 = -a;
        W_92 = W_92+ a *Ghimj[422];
        W_124 = W_124+ a *Ghimj[423];
        W_135 = W_135+ a *Ghimj[424];
        W_137 = W_137+ a *Ghimj[425];
        a = - W_86/ Ghimj[436];
        W_86 = -a;
        W_93 = W_93+ a *Ghimj[437];
        W_125 = W_125+ a *Ghimj[438];
        W_126 = W_126+ a *Ghimj[439];
        W_133 = W_133+ a *Ghimj[440];
        W_137 = W_137+ a *Ghimj[441];
        a = - W_87/ Ghimj[444];
        W_87 = -a;
        W_92 = W_92+ a *Ghimj[445];
        W_124 = W_124+ a *Ghimj[446];
        W_126 = W_126+ a *Ghimj[447];
        W_135 = W_135+ a *Ghimj[448];
        W_137 = W_137+ a *Ghimj[449];
        a = - W_88/ Ghimj[450];
        W_88 = -a;
        W_103 = W_103+ a *Ghimj[451];
        W_106 = W_106+ a *Ghimj[452];
        W_124 = W_124+ a *Ghimj[453];
        W_126 = W_126+ a *Ghimj[454];
        W_127 = W_127+ a *Ghimj[455];
        W_137 = W_137+ a *Ghimj[456];
        a = - W_92/ Ghimj[489];
        W_92 = -a;
        W_124 = W_124+ a *Ghimj[490];
        W_126 = W_126+ a *Ghimj[491];
        W_133 = W_133+ a *Ghimj[492];
        W_135 = W_135+ a *Ghimj[493];
        W_137 = W_137+ a *Ghimj[494];
        a = - W_93/ Ghimj[497];
        W_93 = -a;
        W_125 = W_125+ a *Ghimj[498];
        W_126 = W_126+ a *Ghimj[499];
        W_133 = W_133+ a *Ghimj[500];
        W_137 = W_137+ a *Ghimj[501];
        a = - W_97/ Ghimj[549];
        W_97 = -a;
        W_98 = W_98+ a *Ghimj[550];
        W_120 = W_120+ a *Ghimj[551];
        W_122 = W_122+ a *Ghimj[552];
        W_126 = W_126+ a *Ghimj[553];
        W_127 = W_127+ a *Ghimj[554];
        W_130 = W_130+ a *Ghimj[555];
        W_137 = W_137+ a *Ghimj[556];
        a = - W_98/ Ghimj[557];
        W_98 = -a;
        W_107 = W_107+ a *Ghimj[558];
        W_120 = W_120+ a *Ghimj[559];
        W_124 = W_124+ a *Ghimj[560];
        W_126 = W_126+ a *Ghimj[561];
        W_127 = W_127+ a *Ghimj[562];
        a = - W_101/ Ghimj[586];
        W_101 = -a;
        W_105 = W_105+ a *Ghimj[587];
        W_114 = W_114+ a *Ghimj[588];
        W_116 = W_116+ a *Ghimj[589];
        W_119 = W_119+ a *Ghimj[590];
        W_123 = W_123+ a *Ghimj[591];
        W_126 = W_126+ a *Ghimj[592];
        W_128 = W_128+ a *Ghimj[593];
        W_130 = W_130+ a *Ghimj[594];
        W_135 = W_135+ a *Ghimj[595];
        W_136 = W_136+ a *Ghimj[596];
        W_138 = W_138+ a *Ghimj[597];
        a = - W_102/ Ghimj[600];
        W_102 = -a;
        W_125 = W_125+ a *Ghimj[601];
        W_126 = W_126+ a *Ghimj[602];
        W_133 = W_133+ a *Ghimj[603];
        W_137 = W_137+ a *Ghimj[604];
        a = - W_103/ Ghimj[605];
        W_103 = -a;
        W_124 = W_124+ a *Ghimj[606];
        W_126 = W_126+ a *Ghimj[607];
        W_127 = W_127+ a *Ghimj[608];
        W_129 = W_129+ a *Ghimj[609];
        a = - W_104/ Ghimj[610];
        W_104 = -a;
        W_125 = W_125+ a *Ghimj[611];
        W_126 = W_126+ a *Ghimj[612];
        W_127 = W_127+ a *Ghimj[613];
        W_129 = W_129+ a *Ghimj[614];
        W_137 = W_137+ a *Ghimj[615];
        a = - W_105/ Ghimj[616];
        W_105 = -a;
        W_128 = W_128+ a *Ghimj[617];
        W_129 = W_129+ a *Ghimj[618];
        W_132 = W_132+ a *Ghimj[619];
        W_135 = W_135+ a *Ghimj[620];
        W_138 = W_138+ a *Ghimj[621];
        a = - W_106/ Ghimj[622];
        W_106 = -a;
        W_124 = W_124+ a *Ghimj[623];
        W_126 = W_126+ a *Ghimj[624];
        W_136 = W_136+ a *Ghimj[625];
        a = - W_107/ Ghimj[626];
        W_107 = -a;
        W_124 = W_124+ a *Ghimj[627];
        W_126 = W_126+ a *Ghimj[628];
        W_136 = W_136+ a *Ghimj[629];
        a = - W_110/ Ghimj[659];
        W_110 = -a;
        W_124 = W_124+ a *Ghimj[660];
        W_125 = W_125+ a *Ghimj[661];
        W_126 = W_126+ a *Ghimj[662];
        W_133 = W_133+ a *Ghimj[663];
        W_136 = W_136+ a *Ghimj[664];
        W_137 = W_137+ a *Ghimj[665];
        a = - W_111/ Ghimj[669];
        W_111 = -a;
        W_115 = W_115+ a *Ghimj[670];
        W_124 = W_124+ a *Ghimj[671];
        W_125 = W_125+ a *Ghimj[672];
        W_126 = W_126+ a *Ghimj[673];
        W_133 = W_133+ a *Ghimj[674];
        W_136 = W_136+ a *Ghimj[675];
        W_137 = W_137+ a *Ghimj[676];
        a = - W_112/ Ghimj[677];
        W_112 = -a;
        W_116 = W_116+ a *Ghimj[678];
        W_123 = W_123+ a *Ghimj[679];
        W_126 = W_126+ a *Ghimj[680];
        W_128 = W_128+ a *Ghimj[681];
        W_134 = W_134+ a *Ghimj[682];
        W_137 = W_137+ a *Ghimj[683];
        W_138 = W_138+ a *Ghimj[684];
        a = - W_114/ Ghimj[697];
        W_114 = -a;
        W_126 = W_126+ a *Ghimj[698];
        W_127 = W_127+ a *Ghimj[699];
        W_129 = W_129+ a *Ghimj[700];
        W_132 = W_132+ a *Ghimj[701];
        W_136 = W_136+ a *Ghimj[702];
        a = - W_115/ Ghimj[706];
        W_115 = -a;
        W_124 = W_124+ a *Ghimj[707];
        W_126 = W_126+ a *Ghimj[708];
        W_127 = W_127+ a *Ghimj[709];
        W_129 = W_129+ a *Ghimj[710];
        W_133 = W_133+ a *Ghimj[711];
        W_136 = W_136+ a *Ghimj[712];
        W_137 = W_137+ a *Ghimj[713];
        a = - W_116/ Ghimj[714];
        W_116 = -a;
        W_123 = W_123+ a *Ghimj[715];
        W_127 = W_127+ a *Ghimj[716];
        W_128 = W_128+ a *Ghimj[717];
        W_131 = W_131+ a *Ghimj[718];
        W_134 = W_134+ a *Ghimj[719];
        W_135 = W_135+ a *Ghimj[720];
        W_138 = W_138+ a *Ghimj[721];
        a = - W_117/ Ghimj[731];
        W_117 = -a;
        W_121 = W_121+ a *Ghimj[732];
        W_124 = W_124+ a *Ghimj[733];
        W_125 = W_125+ a *Ghimj[734];
        W_126 = W_126+ a *Ghimj[735];
        W_127 = W_127+ a *Ghimj[736];
        W_129 = W_129+ a *Ghimj[737];
        W_133 = W_133+ a *Ghimj[738];
        W_136 = W_136+ a *Ghimj[739];
        W_137 = W_137+ a *Ghimj[740];
        a = - W_118/ Ghimj[745];
        W_118 = -a;
        W_123 = W_123+ a *Ghimj[746];
        W_125 = W_125+ a *Ghimj[747];
        W_126 = W_126+ a *Ghimj[748];
        W_127 = W_127+ a *Ghimj[749];
        W_128 = W_128+ a *Ghimj[750];
        W_129 = W_129+ a *Ghimj[751];
        W_131 = W_131+ a *Ghimj[752];
        W_132 = W_132+ a *Ghimj[753];
        W_134 = W_134+ a *Ghimj[754];
        W_135 = W_135+ a *Ghimj[755];
        W_137 = W_137+ a *Ghimj[756];
        W_138 = W_138+ a *Ghimj[757];
        a = - W_119/ Ghimj[767];
        W_119 = -a;
        W_121 = W_121+ a *Ghimj[768];
        W_124 = W_124+ a *Ghimj[769];
        W_125 = W_125+ a *Ghimj[770];
        W_126 = W_126+ a *Ghimj[771];
        W_127 = W_127+ a *Ghimj[772];
        W_129 = W_129+ a *Ghimj[773];
        W_133 = W_133+ a *Ghimj[774];
        W_136 = W_136+ a *Ghimj[775];
        W_137 = W_137+ a *Ghimj[776];
        a = - W_120/ Ghimj[787];
        W_120 = -a;
        W_122 = W_122+ a *Ghimj[788];
        W_124 = W_124+ a *Ghimj[789];
        W_126 = W_126+ a *Ghimj[790];
        W_127 = W_127+ a *Ghimj[791];
        W_128 = W_128+ a *Ghimj[792];
        W_130 = W_130+ a *Ghimj[793];
        W_133 = W_133+ a *Ghimj[794];
        W_135 = W_135+ a *Ghimj[795];
        W_136 = W_136+ a *Ghimj[796];
        W_137 = W_137+ a *Ghimj[797];
        a = - W_121/ Ghimj[821];
        W_121 = -a;
        W_124 = W_124+ a *Ghimj[822];
        W_125 = W_125+ a *Ghimj[823];
        W_126 = W_126+ a *Ghimj[824];
        W_127 = W_127+ a *Ghimj[825];
        W_129 = W_129+ a *Ghimj[826];
        W_133 = W_133+ a *Ghimj[827];
        W_135 = W_135+ a *Ghimj[828];
        W_136 = W_136+ a *Ghimj[829];
        W_137 = W_137+ a *Ghimj[830];
        a = - W_122/ Ghimj[847];
        W_122 = -a;
        W_124 = W_124+ a *Ghimj[848];
        W_125 = W_125+ a *Ghimj[849];
        W_126 = W_126+ a *Ghimj[850];
        W_127 = W_127+ a *Ghimj[851];
        W_128 = W_128+ a *Ghimj[852];
        W_129 = W_129+ a *Ghimj[853];
        W_130 = W_130+ a *Ghimj[854];
        W_131 = W_131+ a *Ghimj[855];
        W_133 = W_133+ a *Ghimj[856];
        W_135 = W_135+ a *Ghimj[857];
        W_136 = W_136+ a *Ghimj[858];
        W_137 = W_137+ a *Ghimj[859];
        W_138 = W_138+ a *Ghimj[860];
        a = - W_123/ Ghimj[869];
        W_123 = -a;
        W_124 = W_124+ a *Ghimj[870];
        W_125 = W_125+ a *Ghimj[871];
        W_126 = W_126+ a *Ghimj[872];
        W_127 = W_127+ a *Ghimj[873];
        W_128 = W_128+ a *Ghimj[874];
        W_129 = W_129+ a *Ghimj[875];
        W_130 = W_130+ a *Ghimj[876];
        W_131 = W_131+ a *Ghimj[877];
        W_132 = W_132+ a *Ghimj[878];
        W_133 = W_133+ a *Ghimj[879];
        W_134 = W_134+ a *Ghimj[880];
        W_135 = W_135+ a *Ghimj[881];
        W_136 = W_136+ a *Ghimj[882];
        W_137 = W_137+ a *Ghimj[883];
        W_138 = W_138+ a *Ghimj[884];
        a = - W_124/ Ghimj[896];
        W_124 = -a;
        W_125 = W_125+ a *Ghimj[897];
        W_126 = W_126+ a *Ghimj[898];
        W_127 = W_127+ a *Ghimj[899];
        W_128 = W_128+ a *Ghimj[900];
        W_129 = W_129+ a *Ghimj[901];
        W_130 = W_130+ a *Ghimj[902];
        W_131 = W_131+ a *Ghimj[903];
        W_132 = W_132+ a *Ghimj[904];
        W_133 = W_133+ a *Ghimj[905];
        W_135 = W_135+ a *Ghimj[906];
        W_136 = W_136+ a *Ghimj[907];
        W_137 = W_137+ a *Ghimj[908];
        W_138 = W_138+ a *Ghimj[909];
        a = - W_125/ Ghimj[934];
        W_125 = -a;
        W_126 = W_126+ a *Ghimj[935];
        W_127 = W_127+ a *Ghimj[936];
        W_128 = W_128+ a *Ghimj[937];
        W_129 = W_129+ a *Ghimj[938];
        W_130 = W_130+ a *Ghimj[939];
        W_131 = W_131+ a *Ghimj[940];
        W_132 = W_132+ a *Ghimj[941];
        W_133 = W_133+ a *Ghimj[942];
        W_134 = W_134+ a *Ghimj[943];
        W_135 = W_135+ a *Ghimj[944];
        W_136 = W_136+ a *Ghimj[945];
        W_137 = W_137+ a *Ghimj[946];
        W_138 = W_138+ a *Ghimj[947];
        a = - W_126/ Ghimj[1023];
        W_126 = -a;
        W_127 = W_127+ a *Ghimj[1024];
        W_128 = W_128+ a *Ghimj[1025];
        W_129 = W_129+ a *Ghimj[1026];
        W_130 = W_130+ a *Ghimj[1027];
        W_131 = W_131+ a *Ghimj[1028];
        W_132 = W_132+ a *Ghimj[1029];
        W_133 = W_133+ a *Ghimj[1030];
        W_134 = W_134+ a *Ghimj[1031];
        W_135 = W_135+ a *Ghimj[1032];
        W_136 = W_136+ a *Ghimj[1033];
        W_137 = W_137+ a *Ghimj[1034];
        W_138 = W_138+ a *Ghimj[1035];
        a = - W_127/ Ghimj[1071];
        W_127 = -a;
        W_128 = W_128+ a *Ghimj[1072];
        W_129 = W_129+ a *Ghimj[1073];
        W_130 = W_130+ a *Ghimj[1074];
        W_131 = W_131+ a *Ghimj[1075];
        W_132 = W_132+ a *Ghimj[1076];
        W_133 = W_133+ a *Ghimj[1077];
        W_134 = W_134+ a *Ghimj[1078];
        W_135 = W_135+ a *Ghimj[1079];
        W_136 = W_136+ a *Ghimj[1080];
        W_137 = W_137+ a *Ghimj[1081];
        W_138 = W_138+ a *Ghimj[1082];
        Ghimj[1083] = W_40;
        Ghimj[1084] = W_44;
        Ghimj[1085] = W_45;
        Ghimj[1086] = W_47;
        Ghimj[1087] = W_48;
        Ghimj[1088] = W_49;
        Ghimj[1089] = W_52;
        Ghimj[1090] = W_53;
        Ghimj[1091] = W_54;
        Ghimj[1092] = W_55;
        Ghimj[1093] = W_57;
        Ghimj[1094] = W_61;
        Ghimj[1095] = W_63;
        Ghimj[1096] = W_67;
        Ghimj[1097] = W_70;
        Ghimj[1098] = W_73;
        Ghimj[1099] = W_74;
        Ghimj[1100] = W_75;
        Ghimj[1101] = W_76;
        Ghimj[1102] = W_77;
        Ghimj[1103] = W_78;
        Ghimj[1104] = W_79;
        Ghimj[1105] = W_83;
        Ghimj[1106] = W_84;
        Ghimj[1107] = W_86;
        Ghimj[1108] = W_87;
        Ghimj[1109] = W_88;
        Ghimj[1110] = W_92;
        Ghimj[1111] = W_93;
        Ghimj[1112] = W_97;
        Ghimj[1113] = W_98;
        Ghimj[1114] = W_101;
        Ghimj[1115] = W_102;
        Ghimj[1116] = W_103;
        Ghimj[1117] = W_104;
        Ghimj[1118] = W_105;
        Ghimj[1119] = W_106;
        Ghimj[1120] = W_107;
        Ghimj[1121] = W_110;
        Ghimj[1122] = W_111;
        Ghimj[1123] = W_112;
        Ghimj[1124] = W_114;
        Ghimj[1125] = W_115;
        Ghimj[1126] = W_116;
        Ghimj[1127] = W_117;
        Ghimj[1128] = W_118;
        Ghimj[1129] = W_119;
        Ghimj[1130] = W_120;
        Ghimj[1131] = W_121;
        Ghimj[1132] = W_122;
        Ghimj[1133] = W_123;
        Ghimj[1134] = W_124;
        Ghimj[1135] = W_125;
        Ghimj[1136] = W_126;
        Ghimj[1137] = W_127;
        Ghimj[1138] = W_128;
        Ghimj[1139] = W_129;
        Ghimj[1140] = W_130;
        Ghimj[1141] = W_131;
        Ghimj[1142] = W_132;
        Ghimj[1143] = W_133;
        Ghimj[1144] = W_134;
        Ghimj[1145] = W_135;
        Ghimj[1146] = W_136;
        Ghimj[1147] = W_137;
        Ghimj[1148] = W_138;
        W_0 = Ghimj[1149];
        W_1 = Ghimj[1150];
        W_2 = Ghimj[1151];
        W_44 = Ghimj[1152];
        W_45 = Ghimj[1153];
        W_52 = Ghimj[1154];
        W_53 = Ghimj[1155];
        W_54 = Ghimj[1156];
        W_55 = Ghimj[1157];
        W_80 = Ghimj[1158];
        W_90 = Ghimj[1159];
        W_100 = Ghimj[1160];
        W_103 = Ghimj[1161];
        W_104 = Ghimj[1162];
        W_105 = Ghimj[1163];
        W_112 = Ghimj[1164];
        W_114 = Ghimj[1165];
        W_116 = Ghimj[1166];
        W_118 = Ghimj[1167];
        W_119 = Ghimj[1168];
        W_121 = Ghimj[1169];
        W_123 = Ghimj[1170];
        W_124 = Ghimj[1171];
        W_125 = Ghimj[1172];
        W_126 = Ghimj[1173];
        W_127 = Ghimj[1174];
        W_128 = Ghimj[1175];
        W_129 = Ghimj[1176];
        W_130 = Ghimj[1177];
        W_131 = Ghimj[1178];
        W_132 = Ghimj[1179];
        W_133 = Ghimj[1180];
        W_134 = Ghimj[1181];
        W_135 = Ghimj[1182];
        W_136 = Ghimj[1183];
        W_137 = Ghimj[1184];
        W_138 = Ghimj[1185];
        a = - W_0/ Ghimj[0];
        W_0 = -a;
        a = - W_1/ Ghimj[1];
        W_1 = -a;
        a = - W_2/ Ghimj[2];
        W_2 = -a;
        a = - W_44/ Ghimj[268];
        W_44 = -a;
        W_126 = W_126+ a *Ghimj[269];
        a = - W_45/ Ghimj[270];
        W_45 = -a;
        W_126 = W_126+ a *Ghimj[271];
        a = - W_52/ Ghimj[288];
        W_52 = -a;
        W_126 = W_126+ a *Ghimj[289];
        a = - W_53/ Ghimj[290];
        W_53 = -a;
        W_126 = W_126+ a *Ghimj[291];
        a = - W_54/ Ghimj[292];
        W_54 = -a;
        W_126 = W_126+ a *Ghimj[293];
        a = - W_55/ Ghimj[294];
        W_55 = -a;
        W_126 = W_126+ a *Ghimj[295];
        a = - W_80/ Ghimj[397];
        W_80 = -a;
        W_90 = W_90+ a *Ghimj[398];
        W_112 = W_112+ a *Ghimj[399];
        W_116 = W_116+ a *Ghimj[400];
        W_127 = W_127+ a *Ghimj[401];
        W_129 = W_129+ a *Ghimj[402];
        W_134 = W_134+ a *Ghimj[403];
        W_138 = W_138+ a *Ghimj[404];
        a = - W_90/ Ghimj[469];
        W_90 = -a;
        W_100 = W_100+ a *Ghimj[470];
        W_105 = W_105+ a *Ghimj[471];
        W_112 = W_112+ a *Ghimj[472];
        W_116 = W_116+ a *Ghimj[473];
        W_118 = W_118+ a *Ghimj[474];
        W_123 = W_123+ a *Ghimj[475];
        W_127 = W_127+ a *Ghimj[476];
        W_129 = W_129+ a *Ghimj[477];
        W_132 = W_132+ a *Ghimj[478];
        W_134 = W_134+ a *Ghimj[479];
        W_138 = W_138+ a *Ghimj[480];
        a = - W_100/ Ghimj[573];
        W_100 = -a;
        W_105 = W_105+ a *Ghimj[574];
        W_112 = W_112+ a *Ghimj[575];
        W_116 = W_116+ a *Ghimj[576];
        W_118 = W_118+ a *Ghimj[577];
        W_123 = W_123+ a *Ghimj[578];
        W_126 = W_126+ a *Ghimj[579];
        W_127 = W_127+ a *Ghimj[580];
        W_129 = W_129+ a *Ghimj[581];
        W_132 = W_132+ a *Ghimj[582];
        W_134 = W_134+ a *Ghimj[583];
        W_138 = W_138+ a *Ghimj[584];
        a = - W_103/ Ghimj[605];
        W_103 = -a;
        W_124 = W_124+ a *Ghimj[606];
        W_126 = W_126+ a *Ghimj[607];
        W_127 = W_127+ a *Ghimj[608];
        W_129 = W_129+ a *Ghimj[609];
        a = - W_104/ Ghimj[610];
        W_104 = -a;
        W_125 = W_125+ a *Ghimj[611];
        W_126 = W_126+ a *Ghimj[612];
        W_127 = W_127+ a *Ghimj[613];
        W_129 = W_129+ a *Ghimj[614];
        W_137 = W_137+ a *Ghimj[615];
        a = - W_105/ Ghimj[616];
        W_105 = -a;
        W_128 = W_128+ a *Ghimj[617];
        W_129 = W_129+ a *Ghimj[618];
        W_132 = W_132+ a *Ghimj[619];
        W_135 = W_135+ a *Ghimj[620];
        W_138 = W_138+ a *Ghimj[621];
        a = - W_112/ Ghimj[677];
        W_112 = -a;
        W_116 = W_116+ a *Ghimj[678];
        W_123 = W_123+ a *Ghimj[679];
        W_126 = W_126+ a *Ghimj[680];
        W_128 = W_128+ a *Ghimj[681];
        W_134 = W_134+ a *Ghimj[682];
        W_137 = W_137+ a *Ghimj[683];
        W_138 = W_138+ a *Ghimj[684];
        a = - W_114/ Ghimj[697];
        W_114 = -a;
        W_126 = W_126+ a *Ghimj[698];
        W_127 = W_127+ a *Ghimj[699];
        W_129 = W_129+ a *Ghimj[700];
        W_132 = W_132+ a *Ghimj[701];
        W_136 = W_136+ a *Ghimj[702];
        a = - W_116/ Ghimj[714];
        W_116 = -a;
        W_123 = W_123+ a *Ghimj[715];
        W_127 = W_127+ a *Ghimj[716];
        W_128 = W_128+ a *Ghimj[717];
        W_131 = W_131+ a *Ghimj[718];
        W_134 = W_134+ a *Ghimj[719];
        W_135 = W_135+ a *Ghimj[720];
        W_138 = W_138+ a *Ghimj[721];
        a = - W_118/ Ghimj[745];
        W_118 = -a;
        W_123 = W_123+ a *Ghimj[746];
        W_125 = W_125+ a *Ghimj[747];
        W_126 = W_126+ a *Ghimj[748];
        W_127 = W_127+ a *Ghimj[749];
        W_128 = W_128+ a *Ghimj[750];
        W_129 = W_129+ a *Ghimj[751];
        W_131 = W_131+ a *Ghimj[752];
        W_132 = W_132+ a *Ghimj[753];
        W_134 = W_134+ a *Ghimj[754];
        W_135 = W_135+ a *Ghimj[755];
        W_137 = W_137+ a *Ghimj[756];
        W_138 = W_138+ a *Ghimj[757];
        a = - W_119/ Ghimj[767];
        W_119 = -a;
        W_121 = W_121+ a *Ghimj[768];
        W_124 = W_124+ a *Ghimj[769];
        W_125 = W_125+ a *Ghimj[770];
        W_126 = W_126+ a *Ghimj[771];
        W_127 = W_127+ a *Ghimj[772];
        W_129 = W_129+ a *Ghimj[773];
        W_133 = W_133+ a *Ghimj[774];
        W_136 = W_136+ a *Ghimj[775];
        W_137 = W_137+ a *Ghimj[776];
        a = - W_121/ Ghimj[821];
        W_121 = -a;
        W_124 = W_124+ a *Ghimj[822];
        W_125 = W_125+ a *Ghimj[823];
        W_126 = W_126+ a *Ghimj[824];
        W_127 = W_127+ a *Ghimj[825];
        W_129 = W_129+ a *Ghimj[826];
        W_133 = W_133+ a *Ghimj[827];
        W_135 = W_135+ a *Ghimj[828];
        W_136 = W_136+ a *Ghimj[829];
        W_137 = W_137+ a *Ghimj[830];
        a = - W_123/ Ghimj[869];
        W_123 = -a;
        W_124 = W_124+ a *Ghimj[870];
        W_125 = W_125+ a *Ghimj[871];
        W_126 = W_126+ a *Ghimj[872];
        W_127 = W_127+ a *Ghimj[873];
        W_128 = W_128+ a *Ghimj[874];
        W_129 = W_129+ a *Ghimj[875];
        W_130 = W_130+ a *Ghimj[876];
        W_131 = W_131+ a *Ghimj[877];
        W_132 = W_132+ a *Ghimj[878];
        W_133 = W_133+ a *Ghimj[879];
        W_134 = W_134+ a *Ghimj[880];
        W_135 = W_135+ a *Ghimj[881];
        W_136 = W_136+ a *Ghimj[882];
        W_137 = W_137+ a *Ghimj[883];
        W_138 = W_138+ a *Ghimj[884];
        a = - W_124/ Ghimj[896];
        W_124 = -a;
        W_125 = W_125+ a *Ghimj[897];
        W_126 = W_126+ a *Ghimj[898];
        W_127 = W_127+ a *Ghimj[899];
        W_128 = W_128+ a *Ghimj[900];
        W_129 = W_129+ a *Ghimj[901];
        W_130 = W_130+ a *Ghimj[902];
        W_131 = W_131+ a *Ghimj[903];
        W_132 = W_132+ a *Ghimj[904];
        W_133 = W_133+ a *Ghimj[905];
        W_135 = W_135+ a *Ghimj[906];
        W_136 = W_136+ a *Ghimj[907];
        W_137 = W_137+ a *Ghimj[908];
        W_138 = W_138+ a *Ghimj[909];
        a = - W_125/ Ghimj[934];
        W_125 = -a;
        W_126 = W_126+ a *Ghimj[935];
        W_127 = W_127+ a *Ghimj[936];
        W_128 = W_128+ a *Ghimj[937];
        W_129 = W_129+ a *Ghimj[938];
        W_130 = W_130+ a *Ghimj[939];
        W_131 = W_131+ a *Ghimj[940];
        W_132 = W_132+ a *Ghimj[941];
        W_133 = W_133+ a *Ghimj[942];
        W_134 = W_134+ a *Ghimj[943];
        W_135 = W_135+ a *Ghimj[944];
        W_136 = W_136+ a *Ghimj[945];
        W_137 = W_137+ a *Ghimj[946];
        W_138 = W_138+ a *Ghimj[947];
        a = - W_126/ Ghimj[1023];
        W_126 = -a;
        W_127 = W_127+ a *Ghimj[1024];
        W_128 = W_128+ a *Ghimj[1025];
        W_129 = W_129+ a *Ghimj[1026];
        W_130 = W_130+ a *Ghimj[1027];
        W_131 = W_131+ a *Ghimj[1028];
        W_132 = W_132+ a *Ghimj[1029];
        W_133 = W_133+ a *Ghimj[1030];
        W_134 = W_134+ a *Ghimj[1031];
        W_135 = W_135+ a *Ghimj[1032];
        W_136 = W_136+ a *Ghimj[1033];
        W_137 = W_137+ a *Ghimj[1034];
        W_138 = W_138+ a *Ghimj[1035];
        a = - W_127/ Ghimj[1071];
        W_127 = -a;
        W_128 = W_128+ a *Ghimj[1072];
        W_129 = W_129+ a *Ghimj[1073];
        W_130 = W_130+ a *Ghimj[1074];
        W_131 = W_131+ a *Ghimj[1075];
        W_132 = W_132+ a *Ghimj[1076];
        W_133 = W_133+ a *Ghimj[1077];
        W_134 = W_134+ a *Ghimj[1078];
        W_135 = W_135+ a *Ghimj[1079];
        W_136 = W_136+ a *Ghimj[1080];
        W_137 = W_137+ a *Ghimj[1081];
        W_138 = W_138+ a *Ghimj[1082];
        a = - W_128/ Ghimj[1138];
        W_128 = -a;
        W_129 = W_129+ a *Ghimj[1139];
        W_130 = W_130+ a *Ghimj[1140];
        W_131 = W_131+ a *Ghimj[1141];
        W_132 = W_132+ a *Ghimj[1142];
        W_133 = W_133+ a *Ghimj[1143];
        W_134 = W_134+ a *Ghimj[1144];
        W_135 = W_135+ a *Ghimj[1145];
        W_136 = W_136+ a *Ghimj[1146];
        W_137 = W_137+ a *Ghimj[1147];
        W_138 = W_138+ a *Ghimj[1148];
        Ghimj[1149] = W_0;
        Ghimj[1150] = W_1;
        Ghimj[1151] = W_2;
        Ghimj[1152] = W_44;
        Ghimj[1153] = W_45;
        Ghimj[1154] = W_52;
        Ghimj[1155] = W_53;
        Ghimj[1156] = W_54;
        Ghimj[1157] = W_55;
        Ghimj[1158] = W_80;
        Ghimj[1159] = W_90;
        Ghimj[1160] = W_100;
        Ghimj[1161] = W_103;
        Ghimj[1162] = W_104;
        Ghimj[1163] = W_105;
        Ghimj[1164] = W_112;
        Ghimj[1165] = W_114;
        Ghimj[1166] = W_116;
        Ghimj[1167] = W_118;
        Ghimj[1168] = W_119;
        Ghimj[1169] = W_121;
        Ghimj[1170] = W_123;
        Ghimj[1171] = W_124;
        Ghimj[1172] = W_125;
        Ghimj[1173] = W_126;
        Ghimj[1174] = W_127;
        Ghimj[1175] = W_128;
        Ghimj[1176] = W_129;
        Ghimj[1177] = W_130;
        Ghimj[1178] = W_131;
        Ghimj[1179] = W_132;
        Ghimj[1180] = W_133;
        Ghimj[1181] = W_134;
        Ghimj[1182] = W_135;
        Ghimj[1183] = W_136;
        Ghimj[1184] = W_137;
        Ghimj[1185] = W_138;
        W_58 = Ghimj[1186];
        W_65 = Ghimj[1187];
        W_66 = Ghimj[1188];
        W_72 = Ghimj[1189];
        W_77 = Ghimj[1190];
        W_82 = Ghimj[1191];
        W_89 = Ghimj[1192];
        W_91 = Ghimj[1193];
        W_93 = Ghimj[1194];
        W_94 = Ghimj[1195];
        W_98 = Ghimj[1196];
        W_102 = Ghimj[1197];
        W_103 = Ghimj[1198];
        W_104 = Ghimj[1199];
        W_106 = Ghimj[1200];
        W_107 = Ghimj[1201];
        W_108 = Ghimj[1202];
        W_109 = Ghimj[1203];
        W_110 = Ghimj[1204];
        W_113 = Ghimj[1205];
        W_114 = Ghimj[1206];
        W_115 = Ghimj[1207];
        W_117 = Ghimj[1208];
        W_120 = Ghimj[1209];
        W_121 = Ghimj[1210];
        W_122 = Ghimj[1211];
        W_124 = Ghimj[1212];
        W_125 = Ghimj[1213];
        W_126 = Ghimj[1214];
        W_127 = Ghimj[1215];
        W_128 = Ghimj[1216];
        W_129 = Ghimj[1217];
        W_130 = Ghimj[1218];
        W_131 = Ghimj[1219];
        W_132 = Ghimj[1220];
        W_133 = Ghimj[1221];
        W_134 = Ghimj[1222];
        W_135 = Ghimj[1223];
        W_136 = Ghimj[1224];
        W_137 = Ghimj[1225];
        W_138 = Ghimj[1226];
        a = - W_58/ Ghimj[303];
        W_58 = -a;
        W_91 = W_91+ a *Ghimj[304];
        W_126 = W_126+ a *Ghimj[305];
        a = - W_65/ Ghimj[331];
        W_65 = -a;
        W_114 = W_114+ a *Ghimj[332];
        W_126 = W_126+ a *Ghimj[333];
        W_132 = W_132+ a *Ghimj[334];
        a = - W_66/ Ghimj[335];
        W_66 = -a;
        W_109 = W_109+ a *Ghimj[336];
        W_126 = W_126+ a *Ghimj[337];
        W_137 = W_137+ a *Ghimj[338];
        a = - W_72/ Ghimj[360];
        W_72 = -a;
        W_94 = W_94+ a *Ghimj[361];
        W_126 = W_126+ a *Ghimj[362];
        W_137 = W_137+ a *Ghimj[363];
        a = - W_77/ Ghimj[382];
        W_77 = -a;
        W_121 = W_121+ a *Ghimj[383];
        W_126 = W_126+ a *Ghimj[384];
        W_135 = W_135+ a *Ghimj[385];
        a = - W_82/ Ghimj[412];
        W_82 = -a;
        W_113 = W_113+ a *Ghimj[413];
        W_126 = W_126+ a *Ghimj[414];
        W_137 = W_137+ a *Ghimj[415];
        a = - W_89/ Ghimj[457];
        W_89 = -a;
        W_93 = W_93+ a *Ghimj[458];
        W_94 = W_94+ a *Ghimj[459];
        W_102 = W_102+ a *Ghimj[460];
        W_107 = W_107+ a *Ghimj[461];
        W_109 = W_109+ a *Ghimj[462];
        W_113 = W_113+ a *Ghimj[463];
        W_117 = W_117+ a *Ghimj[464];
        W_124 = W_124+ a *Ghimj[465];
        W_125 = W_125+ a *Ghimj[466];
        W_126 = W_126+ a *Ghimj[467];
        a = - W_91/ Ghimj[481];
        W_91 = -a;
        W_106 = W_106+ a *Ghimj[482];
        W_109 = W_109+ a *Ghimj[483];
        W_126 = W_126+ a *Ghimj[484];
        W_133 = W_133+ a *Ghimj[485];
        W_136 = W_136+ a *Ghimj[486];
        a = - W_93/ Ghimj[497];
        W_93 = -a;
        W_125 = W_125+ a *Ghimj[498];
        W_126 = W_126+ a *Ghimj[499];
        W_133 = W_133+ a *Ghimj[500];
        W_137 = W_137+ a *Ghimj[501];
        a = - W_94/ Ghimj[505];
        W_94 = -a;
        W_125 = W_125+ a *Ghimj[506];
        W_126 = W_126+ a *Ghimj[507];
        W_133 = W_133+ a *Ghimj[508];
        W_137 = W_137+ a *Ghimj[509];
        a = - W_98/ Ghimj[557];
        W_98 = -a;
        W_107 = W_107+ a *Ghimj[558];
        W_120 = W_120+ a *Ghimj[559];
        W_124 = W_124+ a *Ghimj[560];
        W_126 = W_126+ a *Ghimj[561];
        W_127 = W_127+ a *Ghimj[562];
        a = - W_102/ Ghimj[600];
        W_102 = -a;
        W_125 = W_125+ a *Ghimj[601];
        W_126 = W_126+ a *Ghimj[602];
        W_133 = W_133+ a *Ghimj[603];
        W_137 = W_137+ a *Ghimj[604];
        a = - W_103/ Ghimj[605];
        W_103 = -a;
        W_124 = W_124+ a *Ghimj[606];
        W_126 = W_126+ a *Ghimj[607];
        W_127 = W_127+ a *Ghimj[608];
        W_129 = W_129+ a *Ghimj[609];
        a = - W_104/ Ghimj[610];
        W_104 = -a;
        W_125 = W_125+ a *Ghimj[611];
        W_126 = W_126+ a *Ghimj[612];
        W_127 = W_127+ a *Ghimj[613];
        W_129 = W_129+ a *Ghimj[614];
        W_137 = W_137+ a *Ghimj[615];
        a = - W_106/ Ghimj[622];
        W_106 = -a;
        W_124 = W_124+ a *Ghimj[623];
        W_126 = W_126+ a *Ghimj[624];
        W_136 = W_136+ a *Ghimj[625];
        a = - W_107/ Ghimj[626];
        W_107 = -a;
        W_124 = W_124+ a *Ghimj[627];
        W_126 = W_126+ a *Ghimj[628];
        W_136 = W_136+ a *Ghimj[629];
        a = - W_108/ Ghimj[636];
        W_108 = -a;
        W_109 = W_109+ a *Ghimj[637];
        W_113 = W_113+ a *Ghimj[638];
        W_115 = W_115+ a *Ghimj[639];
        W_124 = W_124+ a *Ghimj[640];
        W_125 = W_125+ a *Ghimj[641];
        W_126 = W_126+ a *Ghimj[642];
        W_133 = W_133+ a *Ghimj[643];
        W_135 = W_135+ a *Ghimj[644];
        W_136 = W_136+ a *Ghimj[645];
        W_137 = W_137+ a *Ghimj[646];
        a = - W_109/ Ghimj[648];
        W_109 = -a;
        W_124 = W_124+ a *Ghimj[649];
        W_125 = W_125+ a *Ghimj[650];
        W_126 = W_126+ a *Ghimj[651];
        W_133 = W_133+ a *Ghimj[652];
        W_136 = W_136+ a *Ghimj[653];
        W_137 = W_137+ a *Ghimj[654];
        a = - W_110/ Ghimj[659];
        W_110 = -a;
        W_124 = W_124+ a *Ghimj[660];
        W_125 = W_125+ a *Ghimj[661];
        W_126 = W_126+ a *Ghimj[662];
        W_133 = W_133+ a *Ghimj[663];
        W_136 = W_136+ a *Ghimj[664];
        W_137 = W_137+ a *Ghimj[665];
        a = - W_113/ Ghimj[689];
        W_113 = -a;
        W_124 = W_124+ a *Ghimj[690];
        W_125 = W_125+ a *Ghimj[691];
        W_126 = W_126+ a *Ghimj[692];
        W_133 = W_133+ a *Ghimj[693];
        W_135 = W_135+ a *Ghimj[694];
        W_136 = W_136+ a *Ghimj[695];
        W_137 = W_137+ a *Ghimj[696];
        a = - W_114/ Ghimj[697];
        W_114 = -a;
        W_126 = W_126+ a *Ghimj[698];
        W_127 = W_127+ a *Ghimj[699];
        W_129 = W_129+ a *Ghimj[700];
        W_132 = W_132+ a *Ghimj[701];
        W_136 = W_136+ a *Ghimj[702];
        a = - W_115/ Ghimj[706];
        W_115 = -a;
        W_124 = W_124+ a *Ghimj[707];
        W_126 = W_126+ a *Ghimj[708];
        W_127 = W_127+ a *Ghimj[709];
        W_129 = W_129+ a *Ghimj[710];
        W_133 = W_133+ a *Ghimj[711];
        W_136 = W_136+ a *Ghimj[712];
        W_137 = W_137+ a *Ghimj[713];
        a = - W_117/ Ghimj[731];
        W_117 = -a;
        W_121 = W_121+ a *Ghimj[732];
        W_124 = W_124+ a *Ghimj[733];
        W_125 = W_125+ a *Ghimj[734];
        W_126 = W_126+ a *Ghimj[735];
        W_127 = W_127+ a *Ghimj[736];
        W_129 = W_129+ a *Ghimj[737];
        W_133 = W_133+ a *Ghimj[738];
        W_136 = W_136+ a *Ghimj[739];
        W_137 = W_137+ a *Ghimj[740];
        a = - W_120/ Ghimj[787];
        W_120 = -a;
        W_122 = W_122+ a *Ghimj[788];
        W_124 = W_124+ a *Ghimj[789];
        W_126 = W_126+ a *Ghimj[790];
        W_127 = W_127+ a *Ghimj[791];
        W_128 = W_128+ a *Ghimj[792];
        W_130 = W_130+ a *Ghimj[793];
        W_133 = W_133+ a *Ghimj[794];
        W_135 = W_135+ a *Ghimj[795];
        W_136 = W_136+ a *Ghimj[796];
        W_137 = W_137+ a *Ghimj[797];
        a = - W_121/ Ghimj[821];
        W_121 = -a;
        W_124 = W_124+ a *Ghimj[822];
        W_125 = W_125+ a *Ghimj[823];
        W_126 = W_126+ a *Ghimj[824];
        W_127 = W_127+ a *Ghimj[825];
        W_129 = W_129+ a *Ghimj[826];
        W_133 = W_133+ a *Ghimj[827];
        W_135 = W_135+ a *Ghimj[828];
        W_136 = W_136+ a *Ghimj[829];
        W_137 = W_137+ a *Ghimj[830];
        a = - W_122/ Ghimj[847];
        W_122 = -a;
        W_124 = W_124+ a *Ghimj[848];
        W_125 = W_125+ a *Ghimj[849];
        W_126 = W_126+ a *Ghimj[850];
        W_127 = W_127+ a *Ghimj[851];
        W_128 = W_128+ a *Ghimj[852];
        W_129 = W_129+ a *Ghimj[853];
        W_130 = W_130+ a *Ghimj[854];
        W_131 = W_131+ a *Ghimj[855];
        W_133 = W_133+ a *Ghimj[856];
        W_135 = W_135+ a *Ghimj[857];
        W_136 = W_136+ a *Ghimj[858];
        W_137 = W_137+ a *Ghimj[859];
        W_138 = W_138+ a *Ghimj[860];
        a = - W_124/ Ghimj[896];
        W_124 = -a;
        W_125 = W_125+ a *Ghimj[897];
        W_126 = W_126+ a *Ghimj[898];
        W_127 = W_127+ a *Ghimj[899];
        W_128 = W_128+ a *Ghimj[900];
        W_129 = W_129+ a *Ghimj[901];
        W_130 = W_130+ a *Ghimj[902];
        W_131 = W_131+ a *Ghimj[903];
        W_132 = W_132+ a *Ghimj[904];
        W_133 = W_133+ a *Ghimj[905];
        W_135 = W_135+ a *Ghimj[906];
        W_136 = W_136+ a *Ghimj[907];
        W_137 = W_137+ a *Ghimj[908];
        W_138 = W_138+ a *Ghimj[909];
        a = - W_125/ Ghimj[934];
        W_125 = -a;
        W_126 = W_126+ a *Ghimj[935];
        W_127 = W_127+ a *Ghimj[936];
        W_128 = W_128+ a *Ghimj[937];
        W_129 = W_129+ a *Ghimj[938];
        W_130 = W_130+ a *Ghimj[939];
        W_131 = W_131+ a *Ghimj[940];
        W_132 = W_132+ a *Ghimj[941];
        W_133 = W_133+ a *Ghimj[942];
        W_134 = W_134+ a *Ghimj[943];
        W_135 = W_135+ a *Ghimj[944];
        W_136 = W_136+ a *Ghimj[945];
        W_137 = W_137+ a *Ghimj[946];
        W_138 = W_138+ a *Ghimj[947];
        a = - W_126/ Ghimj[1023];
        W_126 = -a;
        W_127 = W_127+ a *Ghimj[1024];
        W_128 = W_128+ a *Ghimj[1025];
        W_129 = W_129+ a *Ghimj[1026];
        W_130 = W_130+ a *Ghimj[1027];
        W_131 = W_131+ a *Ghimj[1028];
        W_132 = W_132+ a *Ghimj[1029];
        W_133 = W_133+ a *Ghimj[1030];
        W_134 = W_134+ a *Ghimj[1031];
        W_135 = W_135+ a *Ghimj[1032];
        W_136 = W_136+ a *Ghimj[1033];
        W_137 = W_137+ a *Ghimj[1034];
        W_138 = W_138+ a *Ghimj[1035];
        a = - W_127/ Ghimj[1071];
        W_127 = -a;
        W_128 = W_128+ a *Ghimj[1072];
        W_129 = W_129+ a *Ghimj[1073];
        W_130 = W_130+ a *Ghimj[1074];
        W_131 = W_131+ a *Ghimj[1075];
        W_132 = W_132+ a *Ghimj[1076];
        W_133 = W_133+ a *Ghimj[1077];
        W_134 = W_134+ a *Ghimj[1078];
        W_135 = W_135+ a *Ghimj[1079];
        W_136 = W_136+ a *Ghimj[1080];
        W_137 = W_137+ a *Ghimj[1081];
        W_138 = W_138+ a *Ghimj[1082];
        a = - W_128/ Ghimj[1138];
        W_128 = -a;
        W_129 = W_129+ a *Ghimj[1139];
        W_130 = W_130+ a *Ghimj[1140];
        W_131 = W_131+ a *Ghimj[1141];
        W_132 = W_132+ a *Ghimj[1142];
        W_133 = W_133+ a *Ghimj[1143];
        W_134 = W_134+ a *Ghimj[1144];
        W_135 = W_135+ a *Ghimj[1145];
        W_136 = W_136+ a *Ghimj[1146];
        W_137 = W_137+ a *Ghimj[1147];
        W_138 = W_138+ a *Ghimj[1148];
        a = - W_129/ Ghimj[1176];
        W_129 = -a;
        W_130 = W_130+ a *Ghimj[1177];
        W_131 = W_131+ a *Ghimj[1178];
        W_132 = W_132+ a *Ghimj[1179];
        W_133 = W_133+ a *Ghimj[1180];
        W_134 = W_134+ a *Ghimj[1181];
        W_135 = W_135+ a *Ghimj[1182];
        W_136 = W_136+ a *Ghimj[1183];
        W_137 = W_137+ a *Ghimj[1184];
        W_138 = W_138+ a *Ghimj[1185];
        Ghimj[1186] = W_58;
        Ghimj[1187] = W_65;
        Ghimj[1188] = W_66;
        Ghimj[1189] = W_72;
        Ghimj[1190] = W_77;
        Ghimj[1191] = W_82;
        Ghimj[1192] = W_89;
        Ghimj[1193] = W_91;
        Ghimj[1194] = W_93;
        Ghimj[1195] = W_94;
        Ghimj[1196] = W_98;
        Ghimj[1197] = W_102;
        Ghimj[1198] = W_103;
        Ghimj[1199] = W_104;
        Ghimj[1200] = W_106;
        Ghimj[1201] = W_107;
        Ghimj[1202] = W_108;
        Ghimj[1203] = W_109;
        Ghimj[1204] = W_110;
        Ghimj[1205] = W_113;
        Ghimj[1206] = W_114;
        Ghimj[1207] = W_115;
        Ghimj[1208] = W_117;
        Ghimj[1209] = W_120;
        Ghimj[1210] = W_121;
        Ghimj[1211] = W_122;
        Ghimj[1212] = W_124;
        Ghimj[1213] = W_125;
        Ghimj[1214] = W_126;
        Ghimj[1215] = W_127;
        Ghimj[1216] = W_128;
        Ghimj[1217] = W_129;
        Ghimj[1218] = W_130;
        Ghimj[1219] = W_131;
        Ghimj[1220] = W_132;
        Ghimj[1221] = W_133;
        Ghimj[1222] = W_134;
        Ghimj[1223] = W_135;
        Ghimj[1224] = W_136;
        Ghimj[1225] = W_137;
        Ghimj[1226] = W_138;
        W_51 = Ghimj[1227];
        W_59 = Ghimj[1228];
        W_75 = Ghimj[1229];
        W_116 = Ghimj[1230];
        W_118 = Ghimj[1231];
        W_120 = Ghimj[1232];
        W_122 = Ghimj[1233];
        W_123 = Ghimj[1234];
        W_124 = Ghimj[1235];
        W_125 = Ghimj[1236];
        W_126 = Ghimj[1237];
        W_127 = Ghimj[1238];
        W_128 = Ghimj[1239];
        W_129 = Ghimj[1240];
        W_130 = Ghimj[1241];
        W_131 = Ghimj[1242];
        W_132 = Ghimj[1243];
        W_133 = Ghimj[1244];
        W_134 = Ghimj[1245];
        W_135 = Ghimj[1246];
        W_136 = Ghimj[1247];
        W_137 = Ghimj[1248];
        W_138 = Ghimj[1249];
        a = - W_51/ Ghimj[285];
        W_51 = -a;
        W_132 = W_132+ a *Ghimj[286];
        W_134 = W_134+ a *Ghimj[287];
        a = - W_59/ Ghimj[306];
        W_59 = -a;
        W_133 = W_133+ a *Ghimj[307];
        W_135 = W_135+ a *Ghimj[308];
        a = - W_75/ Ghimj[374];
        W_75 = -a;
        W_120 = W_120+ a *Ghimj[375];
        W_126 = W_126+ a *Ghimj[376];
        a = - W_116/ Ghimj[714];
        W_116 = -a;
        W_123 = W_123+ a *Ghimj[715];
        W_127 = W_127+ a *Ghimj[716];
        W_128 = W_128+ a *Ghimj[717];
        W_131 = W_131+ a *Ghimj[718];
        W_134 = W_134+ a *Ghimj[719];
        W_135 = W_135+ a *Ghimj[720];
        W_138 = W_138+ a *Ghimj[721];
        a = - W_118/ Ghimj[745];
        W_118 = -a;
        W_123 = W_123+ a *Ghimj[746];
        W_125 = W_125+ a *Ghimj[747];
        W_126 = W_126+ a *Ghimj[748];
        W_127 = W_127+ a *Ghimj[749];
        W_128 = W_128+ a *Ghimj[750];
        W_129 = W_129+ a *Ghimj[751];
        W_131 = W_131+ a *Ghimj[752];
        W_132 = W_132+ a *Ghimj[753];
        W_134 = W_134+ a *Ghimj[754];
        W_135 = W_135+ a *Ghimj[755];
        W_137 = W_137+ a *Ghimj[756];
        W_138 = W_138+ a *Ghimj[757];
        a = - W_120/ Ghimj[787];
        W_120 = -a;
        W_122 = W_122+ a *Ghimj[788];
        W_124 = W_124+ a *Ghimj[789];
        W_126 = W_126+ a *Ghimj[790];
        W_127 = W_127+ a *Ghimj[791];
        W_128 = W_128+ a *Ghimj[792];
        W_130 = W_130+ a *Ghimj[793];
        W_133 = W_133+ a *Ghimj[794];
        W_135 = W_135+ a *Ghimj[795];
        W_136 = W_136+ a *Ghimj[796];
        W_137 = W_137+ a *Ghimj[797];
        a = - W_122/ Ghimj[847];
        W_122 = -a;
        W_124 = W_124+ a *Ghimj[848];
        W_125 = W_125+ a *Ghimj[849];
        W_126 = W_126+ a *Ghimj[850];
        W_127 = W_127+ a *Ghimj[851];
        W_128 = W_128+ a *Ghimj[852];
        W_129 = W_129+ a *Ghimj[853];
        W_130 = W_130+ a *Ghimj[854];
        W_131 = W_131+ a *Ghimj[855];
        W_133 = W_133+ a *Ghimj[856];
        W_135 = W_135+ a *Ghimj[857];
        W_136 = W_136+ a *Ghimj[858];
        W_137 = W_137+ a *Ghimj[859];
        W_138 = W_138+ a *Ghimj[860];
        a = - W_123/ Ghimj[869];
        W_123 = -a;
        W_124 = W_124+ a *Ghimj[870];
        W_125 = W_125+ a *Ghimj[871];
        W_126 = W_126+ a *Ghimj[872];
        W_127 = W_127+ a *Ghimj[873];
        W_128 = W_128+ a *Ghimj[874];
        W_129 = W_129+ a *Ghimj[875];
        W_130 = W_130+ a *Ghimj[876];
        W_131 = W_131+ a *Ghimj[877];
        W_132 = W_132+ a *Ghimj[878];
        W_133 = W_133+ a *Ghimj[879];
        W_134 = W_134+ a *Ghimj[880];
        W_135 = W_135+ a *Ghimj[881];
        W_136 = W_136+ a *Ghimj[882];
        W_137 = W_137+ a *Ghimj[883];
        W_138 = W_138+ a *Ghimj[884];
        a = - W_124/ Ghimj[896];
        W_124 = -a;
        W_125 = W_125+ a *Ghimj[897];
        W_126 = W_126+ a *Ghimj[898];
        W_127 = W_127+ a *Ghimj[899];
        W_128 = W_128+ a *Ghimj[900];
        W_129 = W_129+ a *Ghimj[901];
        W_130 = W_130+ a *Ghimj[902];
        W_131 = W_131+ a *Ghimj[903];
        W_132 = W_132+ a *Ghimj[904];
        W_133 = W_133+ a *Ghimj[905];
        W_135 = W_135+ a *Ghimj[906];
        W_136 = W_136+ a *Ghimj[907];
        W_137 = W_137+ a *Ghimj[908];
        W_138 = W_138+ a *Ghimj[909];
        a = - W_125/ Ghimj[934];
        W_125 = -a;
        W_126 = W_126+ a *Ghimj[935];
        W_127 = W_127+ a *Ghimj[936];
        W_128 = W_128+ a *Ghimj[937];
        W_129 = W_129+ a *Ghimj[938];
        W_130 = W_130+ a *Ghimj[939];
        W_131 = W_131+ a *Ghimj[940];
        W_132 = W_132+ a *Ghimj[941];
        W_133 = W_133+ a *Ghimj[942];
        W_134 = W_134+ a *Ghimj[943];
        W_135 = W_135+ a *Ghimj[944];
        W_136 = W_136+ a *Ghimj[945];
        W_137 = W_137+ a *Ghimj[946];
        W_138 = W_138+ a *Ghimj[947];
        a = - W_126/ Ghimj[1023];
        W_126 = -a;
        W_127 = W_127+ a *Ghimj[1024];
        W_128 = W_128+ a *Ghimj[1025];
        W_129 = W_129+ a *Ghimj[1026];
        W_130 = W_130+ a *Ghimj[1027];
        W_131 = W_131+ a *Ghimj[1028];
        W_132 = W_132+ a *Ghimj[1029];
        W_133 = W_133+ a *Ghimj[1030];
        W_134 = W_134+ a *Ghimj[1031];
        W_135 = W_135+ a *Ghimj[1032];
        W_136 = W_136+ a *Ghimj[1033];
        W_137 = W_137+ a *Ghimj[1034];
        W_138 = W_138+ a *Ghimj[1035];
        a = - W_127/ Ghimj[1071];
        W_127 = -a;
        W_128 = W_128+ a *Ghimj[1072];
        W_129 = W_129+ a *Ghimj[1073];
        W_130 = W_130+ a *Ghimj[1074];
        W_131 = W_131+ a *Ghimj[1075];
        W_132 = W_132+ a *Ghimj[1076];
        W_133 = W_133+ a *Ghimj[1077];
        W_134 = W_134+ a *Ghimj[1078];
        W_135 = W_135+ a *Ghimj[1079];
        W_136 = W_136+ a *Ghimj[1080];
        W_137 = W_137+ a *Ghimj[1081];
        W_138 = W_138+ a *Ghimj[1082];
        a = - W_128/ Ghimj[1138];
        W_128 = -a;
        W_129 = W_129+ a *Ghimj[1139];
        W_130 = W_130+ a *Ghimj[1140];
        W_131 = W_131+ a *Ghimj[1141];
        W_132 = W_132+ a *Ghimj[1142];
        W_133 = W_133+ a *Ghimj[1143];
        W_134 = W_134+ a *Ghimj[1144];
        W_135 = W_135+ a *Ghimj[1145];
        W_136 = W_136+ a *Ghimj[1146];
        W_137 = W_137+ a *Ghimj[1147];
        W_138 = W_138+ a *Ghimj[1148];
        a = - W_129/ Ghimj[1176];
        W_129 = -a;
        W_130 = W_130+ a *Ghimj[1177];
        W_131 = W_131+ a *Ghimj[1178];
        W_132 = W_132+ a *Ghimj[1179];
        W_133 = W_133+ a *Ghimj[1180];
        W_134 = W_134+ a *Ghimj[1181];
        W_135 = W_135+ a *Ghimj[1182];
        W_136 = W_136+ a *Ghimj[1183];
        W_137 = W_137+ a *Ghimj[1184];
        W_138 = W_138+ a *Ghimj[1185];
        a = - W_130/ Ghimj[1218];
        W_130 = -a;
        W_131 = W_131+ a *Ghimj[1219];
        W_132 = W_132+ a *Ghimj[1220];
        W_133 = W_133+ a *Ghimj[1221];
        W_134 = W_134+ a *Ghimj[1222];
        W_135 = W_135+ a *Ghimj[1223];
        W_136 = W_136+ a *Ghimj[1224];
        W_137 = W_137+ a *Ghimj[1225];
        W_138 = W_138+ a *Ghimj[1226];
        Ghimj[1227] = W_51;
        Ghimj[1228] = W_59;
        Ghimj[1229] = W_75;
        Ghimj[1230] = W_116;
        Ghimj[1231] = W_118;
        Ghimj[1232] = W_120;
        Ghimj[1233] = W_122;
        Ghimj[1234] = W_123;
        Ghimj[1235] = W_124;
        Ghimj[1236] = W_125;
        Ghimj[1237] = W_126;
        Ghimj[1238] = W_127;
        Ghimj[1239] = W_128;
        Ghimj[1240] = W_129;
        Ghimj[1241] = W_130;
        Ghimj[1242] = W_131;
        Ghimj[1243] = W_132;
        Ghimj[1244] = W_133;
        Ghimj[1245] = W_134;
        Ghimj[1246] = W_135;
        Ghimj[1247] = W_136;
        Ghimj[1248] = W_137;
        Ghimj[1249] = W_138;
        W_105 = Ghimj[1250];
        W_114 = Ghimj[1251];
        W_118 = Ghimj[1252];
        W_123 = Ghimj[1253];
        W_124 = Ghimj[1254];
        W_125 = Ghimj[1255];
        W_126 = Ghimj[1256];
        W_127 = Ghimj[1257];
        W_128 = Ghimj[1258];
        W_129 = Ghimj[1259];
        W_130 = Ghimj[1260];
        W_131 = Ghimj[1261];
        W_132 = Ghimj[1262];
        W_133 = Ghimj[1263];
        W_134 = Ghimj[1264];
        W_135 = Ghimj[1265];
        W_136 = Ghimj[1266];
        W_137 = Ghimj[1267];
        W_138 = Ghimj[1268];
        a = - W_105/ Ghimj[616];
        W_105 = -a;
        W_128 = W_128+ a *Ghimj[617];
        W_129 = W_129+ a *Ghimj[618];
        W_132 = W_132+ a *Ghimj[619];
        W_135 = W_135+ a *Ghimj[620];
        W_138 = W_138+ a *Ghimj[621];
        a = - W_114/ Ghimj[697];
        W_114 = -a;
        W_126 = W_126+ a *Ghimj[698];
        W_127 = W_127+ a *Ghimj[699];
        W_129 = W_129+ a *Ghimj[700];
        W_132 = W_132+ a *Ghimj[701];
        W_136 = W_136+ a *Ghimj[702];
        a = - W_118/ Ghimj[745];
        W_118 = -a;
        W_123 = W_123+ a *Ghimj[746];
        W_125 = W_125+ a *Ghimj[747];
        W_126 = W_126+ a *Ghimj[748];
        W_127 = W_127+ a *Ghimj[749];
        W_128 = W_128+ a *Ghimj[750];
        W_129 = W_129+ a *Ghimj[751];
        W_131 = W_131+ a *Ghimj[752];
        W_132 = W_132+ a *Ghimj[753];
        W_134 = W_134+ a *Ghimj[754];
        W_135 = W_135+ a *Ghimj[755];
        W_137 = W_137+ a *Ghimj[756];
        W_138 = W_138+ a *Ghimj[757];
        a = - W_123/ Ghimj[869];
        W_123 = -a;
        W_124 = W_124+ a *Ghimj[870];
        W_125 = W_125+ a *Ghimj[871];
        W_126 = W_126+ a *Ghimj[872];
        W_127 = W_127+ a *Ghimj[873];
        W_128 = W_128+ a *Ghimj[874];
        W_129 = W_129+ a *Ghimj[875];
        W_130 = W_130+ a *Ghimj[876];
        W_131 = W_131+ a *Ghimj[877];
        W_132 = W_132+ a *Ghimj[878];
        W_133 = W_133+ a *Ghimj[879];
        W_134 = W_134+ a *Ghimj[880];
        W_135 = W_135+ a *Ghimj[881];
        W_136 = W_136+ a *Ghimj[882];
        W_137 = W_137+ a *Ghimj[883];
        W_138 = W_138+ a *Ghimj[884];
        a = - W_124/ Ghimj[896];
        W_124 = -a;
        W_125 = W_125+ a *Ghimj[897];
        W_126 = W_126+ a *Ghimj[898];
        W_127 = W_127+ a *Ghimj[899];
        W_128 = W_128+ a *Ghimj[900];
        W_129 = W_129+ a *Ghimj[901];
        W_130 = W_130+ a *Ghimj[902];
        W_131 = W_131+ a *Ghimj[903];
        W_132 = W_132+ a *Ghimj[904];
        W_133 = W_133+ a *Ghimj[905];
        W_135 = W_135+ a *Ghimj[906];
        W_136 = W_136+ a *Ghimj[907];
        W_137 = W_137+ a *Ghimj[908];
        W_138 = W_138+ a *Ghimj[909];
        a = - W_125/ Ghimj[934];
        W_125 = -a;
        W_126 = W_126+ a *Ghimj[935];
        W_127 = W_127+ a *Ghimj[936];
        W_128 = W_128+ a *Ghimj[937];
        W_129 = W_129+ a *Ghimj[938];
        W_130 = W_130+ a *Ghimj[939];
        W_131 = W_131+ a *Ghimj[940];
        W_132 = W_132+ a *Ghimj[941];
        W_133 = W_133+ a *Ghimj[942];
        W_134 = W_134+ a *Ghimj[943];
        W_135 = W_135+ a *Ghimj[944];
        W_136 = W_136+ a *Ghimj[945];
        W_137 = W_137+ a *Ghimj[946];
        W_138 = W_138+ a *Ghimj[947];
        a = - W_126/ Ghimj[1023];
        W_126 = -a;
        W_127 = W_127+ a *Ghimj[1024];
        W_128 = W_128+ a *Ghimj[1025];
        W_129 = W_129+ a *Ghimj[1026];
        W_130 = W_130+ a *Ghimj[1027];
        W_131 = W_131+ a *Ghimj[1028];
        W_132 = W_132+ a *Ghimj[1029];
        W_133 = W_133+ a *Ghimj[1030];
        W_134 = W_134+ a *Ghimj[1031];
        W_135 = W_135+ a *Ghimj[1032];
        W_136 = W_136+ a *Ghimj[1033];
        W_137 = W_137+ a *Ghimj[1034];
        W_138 = W_138+ a *Ghimj[1035];
        a = - W_127/ Ghimj[1071];
        W_127 = -a;
        W_128 = W_128+ a *Ghimj[1072];
        W_129 = W_129+ a *Ghimj[1073];
        W_130 = W_130+ a *Ghimj[1074];
        W_131 = W_131+ a *Ghimj[1075];
        W_132 = W_132+ a *Ghimj[1076];
        W_133 = W_133+ a *Ghimj[1077];
        W_134 = W_134+ a *Ghimj[1078];
        W_135 = W_135+ a *Ghimj[1079];
        W_136 = W_136+ a *Ghimj[1080];
        W_137 = W_137+ a *Ghimj[1081];
        W_138 = W_138+ a *Ghimj[1082];
        a = - W_128/ Ghimj[1138];
        W_128 = -a;
        W_129 = W_129+ a *Ghimj[1139];
        W_130 = W_130+ a *Ghimj[1140];
        W_131 = W_131+ a *Ghimj[1141];
        W_132 = W_132+ a *Ghimj[1142];
        W_133 = W_133+ a *Ghimj[1143];
        W_134 = W_134+ a *Ghimj[1144];
        W_135 = W_135+ a *Ghimj[1145];
        W_136 = W_136+ a *Ghimj[1146];
        W_137 = W_137+ a *Ghimj[1147];
        W_138 = W_138+ a *Ghimj[1148];
        a = - W_129/ Ghimj[1176];
        W_129 = -a;
        W_130 = W_130+ a *Ghimj[1177];
        W_131 = W_131+ a *Ghimj[1178];
        W_132 = W_132+ a *Ghimj[1179];
        W_133 = W_133+ a *Ghimj[1180];
        W_134 = W_134+ a *Ghimj[1181];
        W_135 = W_135+ a *Ghimj[1182];
        W_136 = W_136+ a *Ghimj[1183];
        W_137 = W_137+ a *Ghimj[1184];
        W_138 = W_138+ a *Ghimj[1185];
        a = - W_130/ Ghimj[1218];
        W_130 = -a;
        W_131 = W_131+ a *Ghimj[1219];
        W_132 = W_132+ a *Ghimj[1220];
        W_133 = W_133+ a *Ghimj[1221];
        W_134 = W_134+ a *Ghimj[1222];
        W_135 = W_135+ a *Ghimj[1223];
        W_136 = W_136+ a *Ghimj[1224];
        W_137 = W_137+ a *Ghimj[1225];
        W_138 = W_138+ a *Ghimj[1226];
        a = - W_131/ Ghimj[1242];
        W_131 = -a;
        W_132 = W_132+ a *Ghimj[1243];
        W_133 = W_133+ a *Ghimj[1244];
        W_134 = W_134+ a *Ghimj[1245];
        W_135 = W_135+ a *Ghimj[1246];
        W_136 = W_136+ a *Ghimj[1247];
        W_137 = W_137+ a *Ghimj[1248];
        W_138 = W_138+ a *Ghimj[1249];
        Ghimj[1250] = W_105;
        Ghimj[1251] = W_114;
        Ghimj[1252] = W_118;
        Ghimj[1253] = W_123;
        Ghimj[1254] = W_124;
        Ghimj[1255] = W_125;
        Ghimj[1256] = W_126;
        Ghimj[1257] = W_127;
        Ghimj[1258] = W_128;
        Ghimj[1259] = W_129;
        Ghimj[1260] = W_130;
        Ghimj[1261] = W_131;
        Ghimj[1262] = W_132;
        Ghimj[1263] = W_133;
        Ghimj[1264] = W_134;
        Ghimj[1265] = W_135;
        Ghimj[1266] = W_136;
        Ghimj[1267] = W_137;
        Ghimj[1268] = W_138;
        W_59 = Ghimj[1269];
        W_60 = Ghimj[1270];
        W_70 = Ghimj[1271];
        W_76 = Ghimj[1272];
        W_84 = Ghimj[1273];
        W_87 = Ghimj[1274];
        W_92 = Ghimj[1275];
        W_93 = Ghimj[1276];
        W_94 = Ghimj[1277];
        W_99 = Ghimj[1278];
        W_102 = Ghimj[1279];
        W_109 = Ghimj[1280];
        W_111 = Ghimj[1281];
        W_113 = Ghimj[1282];
        W_115 = Ghimj[1283];
        W_117 = Ghimj[1284];
        W_120 = Ghimj[1285];
        W_121 = Ghimj[1286];
        W_122 = Ghimj[1287];
        W_124 = Ghimj[1288];
        W_125 = Ghimj[1289];
        W_126 = Ghimj[1290];
        W_127 = Ghimj[1291];
        W_128 = Ghimj[1292];
        W_129 = Ghimj[1293];
        W_130 = Ghimj[1294];
        W_131 = Ghimj[1295];
        W_132 = Ghimj[1296];
        W_133 = Ghimj[1297];
        W_134 = Ghimj[1298];
        W_135 = Ghimj[1299];
        W_136 = Ghimj[1300];
        W_137 = Ghimj[1301];
        W_138 = Ghimj[1302];
        a = - W_59/ Ghimj[306];
        W_59 = -a;
        W_133 = W_133+ a *Ghimj[307];
        W_135 = W_135+ a *Ghimj[308];
        a = - W_60/ Ghimj[310];
        W_60 = -a;
        W_92 = W_92+ a *Ghimj[311];
        W_120 = W_120+ a *Ghimj[312];
        W_133 = W_133+ a *Ghimj[313];
        W_135 = W_135+ a *Ghimj[314];
        a = - W_70/ Ghimj[352];
        W_70 = -a;
        W_84 = W_84+ a *Ghimj[353];
        W_87 = W_87+ a *Ghimj[354];
        W_126 = W_126+ a *Ghimj[355];
        a = - W_76/ Ghimj[377];
        W_76 = -a;
        W_87 = W_87+ a *Ghimj[378];
        W_126 = W_126+ a *Ghimj[379];
        W_133 = W_133+ a *Ghimj[380];
        W_135 = W_135+ a *Ghimj[381];
        a = - W_84/ Ghimj[421];
        W_84 = -a;
        W_92 = W_92+ a *Ghimj[422];
        W_124 = W_124+ a *Ghimj[423];
        W_135 = W_135+ a *Ghimj[424];
        W_137 = W_137+ a *Ghimj[425];
        a = - W_87/ Ghimj[444];
        W_87 = -a;
        W_92 = W_92+ a *Ghimj[445];
        W_124 = W_124+ a *Ghimj[446];
        W_126 = W_126+ a *Ghimj[447];
        W_135 = W_135+ a *Ghimj[448];
        W_137 = W_137+ a *Ghimj[449];
        a = - W_92/ Ghimj[489];
        W_92 = -a;
        W_124 = W_124+ a *Ghimj[490];
        W_126 = W_126+ a *Ghimj[491];
        W_133 = W_133+ a *Ghimj[492];
        W_135 = W_135+ a *Ghimj[493];
        W_137 = W_137+ a *Ghimj[494];
        a = - W_93/ Ghimj[497];
        W_93 = -a;
        W_125 = W_125+ a *Ghimj[498];
        W_126 = W_126+ a *Ghimj[499];
        W_133 = W_133+ a *Ghimj[500];
        W_137 = W_137+ a *Ghimj[501];
        a = - W_94/ Ghimj[505];
        W_94 = -a;
        W_125 = W_125+ a *Ghimj[506];
        W_126 = W_126+ a *Ghimj[507];
        W_133 = W_133+ a *Ghimj[508];
        W_137 = W_137+ a *Ghimj[509];
        a = - W_99/ Ghimj[565];
        W_99 = -a;
        W_102 = W_102+ a *Ghimj[566];
        W_111 = W_111+ a *Ghimj[567];
        W_125 = W_125+ a *Ghimj[568];
        W_126 = W_126+ a *Ghimj[569];
        W_133 = W_133+ a *Ghimj[570];
        W_137 = W_137+ a *Ghimj[571];
        a = - W_102/ Ghimj[600];
        W_102 = -a;
        W_125 = W_125+ a *Ghimj[601];
        W_126 = W_126+ a *Ghimj[602];
        W_133 = W_133+ a *Ghimj[603];
        W_137 = W_137+ a *Ghimj[604];
        a = - W_109/ Ghimj[648];
        W_109 = -a;
        W_124 = W_124+ a *Ghimj[649];
        W_125 = W_125+ a *Ghimj[650];
        W_126 = W_126+ a *Ghimj[651];
        W_133 = W_133+ a *Ghimj[652];
        W_136 = W_136+ a *Ghimj[653];
        W_137 = W_137+ a *Ghimj[654];
        a = - W_111/ Ghimj[669];
        W_111 = -a;
        W_115 = W_115+ a *Ghimj[670];
        W_124 = W_124+ a *Ghimj[671];
        W_125 = W_125+ a *Ghimj[672];
        W_126 = W_126+ a *Ghimj[673];
        W_133 = W_133+ a *Ghimj[674];
        W_136 = W_136+ a *Ghimj[675];
        W_137 = W_137+ a *Ghimj[676];
        a = - W_113/ Ghimj[689];
        W_113 = -a;
        W_124 = W_124+ a *Ghimj[690];
        W_125 = W_125+ a *Ghimj[691];
        W_126 = W_126+ a *Ghimj[692];
        W_133 = W_133+ a *Ghimj[693];
        W_135 = W_135+ a *Ghimj[694];
        W_136 = W_136+ a *Ghimj[695];
        W_137 = W_137+ a *Ghimj[696];
        a = - W_115/ Ghimj[706];
        W_115 = -a;
        W_124 = W_124+ a *Ghimj[707];
        W_126 = W_126+ a *Ghimj[708];
        W_127 = W_127+ a *Ghimj[709];
        W_129 = W_129+ a *Ghimj[710];
        W_133 = W_133+ a *Ghimj[711];
        W_136 = W_136+ a *Ghimj[712];
        W_137 = W_137+ a *Ghimj[713];
        a = - W_117/ Ghimj[731];
        W_117 = -a;
        W_121 = W_121+ a *Ghimj[732];
        W_124 = W_124+ a *Ghimj[733];
        W_125 = W_125+ a *Ghimj[734];
        W_126 = W_126+ a *Ghimj[735];
        W_127 = W_127+ a *Ghimj[736];
        W_129 = W_129+ a *Ghimj[737];
        W_133 = W_133+ a *Ghimj[738];
        W_136 = W_136+ a *Ghimj[739];
        W_137 = W_137+ a *Ghimj[740];
        a = - W_120/ Ghimj[787];
        W_120 = -a;
        W_122 = W_122+ a *Ghimj[788];
        W_124 = W_124+ a *Ghimj[789];
        W_126 = W_126+ a *Ghimj[790];
        W_127 = W_127+ a *Ghimj[791];
        W_128 = W_128+ a *Ghimj[792];
        W_130 = W_130+ a *Ghimj[793];
        W_133 = W_133+ a *Ghimj[794];
        W_135 = W_135+ a *Ghimj[795];
        W_136 = W_136+ a *Ghimj[796];
        W_137 = W_137+ a *Ghimj[797];
        a = - W_121/ Ghimj[821];
        W_121 = -a;
        W_124 = W_124+ a *Ghimj[822];
        W_125 = W_125+ a *Ghimj[823];
        W_126 = W_126+ a *Ghimj[824];
        W_127 = W_127+ a *Ghimj[825];
        W_129 = W_129+ a *Ghimj[826];
        W_133 = W_133+ a *Ghimj[827];
        W_135 = W_135+ a *Ghimj[828];
        W_136 = W_136+ a *Ghimj[829];
        W_137 = W_137+ a *Ghimj[830];
        a = - W_122/ Ghimj[847];
        W_122 = -a;
        W_124 = W_124+ a *Ghimj[848];
        W_125 = W_125+ a *Ghimj[849];
        W_126 = W_126+ a *Ghimj[850];
        W_127 = W_127+ a *Ghimj[851];
        W_128 = W_128+ a *Ghimj[852];
        W_129 = W_129+ a *Ghimj[853];
        W_130 = W_130+ a *Ghimj[854];
        W_131 = W_131+ a *Ghimj[855];
        W_133 = W_133+ a *Ghimj[856];
        W_135 = W_135+ a *Ghimj[857];
        W_136 = W_136+ a *Ghimj[858];
        W_137 = W_137+ a *Ghimj[859];
        W_138 = W_138+ a *Ghimj[860];
        a = - W_124/ Ghimj[896];
        W_124 = -a;
        W_125 = W_125+ a *Ghimj[897];
        W_126 = W_126+ a *Ghimj[898];
        W_127 = W_127+ a *Ghimj[899];
        W_128 = W_128+ a *Ghimj[900];
        W_129 = W_129+ a *Ghimj[901];
        W_130 = W_130+ a *Ghimj[902];
        W_131 = W_131+ a *Ghimj[903];
        W_132 = W_132+ a *Ghimj[904];
        W_133 = W_133+ a *Ghimj[905];
        W_135 = W_135+ a *Ghimj[906];
        W_136 = W_136+ a *Ghimj[907];
        W_137 = W_137+ a *Ghimj[908];
        W_138 = W_138+ a *Ghimj[909];
        a = - W_125/ Ghimj[934];
        W_125 = -a;
        W_126 = W_126+ a *Ghimj[935];
        W_127 = W_127+ a *Ghimj[936];
        W_128 = W_128+ a *Ghimj[937];
        W_129 = W_129+ a *Ghimj[938];
        W_130 = W_130+ a *Ghimj[939];
        W_131 = W_131+ a *Ghimj[940];
        W_132 = W_132+ a *Ghimj[941];
        W_133 = W_133+ a *Ghimj[942];
        W_134 = W_134+ a *Ghimj[943];
        W_135 = W_135+ a *Ghimj[944];
        W_136 = W_136+ a *Ghimj[945];
        W_137 = W_137+ a *Ghimj[946];
        W_138 = W_138+ a *Ghimj[947];
        a = - W_126/ Ghimj[1023];
        W_126 = -a;
        W_127 = W_127+ a *Ghimj[1024];
        W_128 = W_128+ a *Ghimj[1025];
        W_129 = W_129+ a *Ghimj[1026];
        W_130 = W_130+ a *Ghimj[1027];
        W_131 = W_131+ a *Ghimj[1028];
        W_132 = W_132+ a *Ghimj[1029];
        W_133 = W_133+ a *Ghimj[1030];
        W_134 = W_134+ a *Ghimj[1031];
        W_135 = W_135+ a *Ghimj[1032];
        W_136 = W_136+ a *Ghimj[1033];
        W_137 = W_137+ a *Ghimj[1034];
        W_138 = W_138+ a *Ghimj[1035];
        a = - W_127/ Ghimj[1071];
        W_127 = -a;
        W_128 = W_128+ a *Ghimj[1072];
        W_129 = W_129+ a *Ghimj[1073];
        W_130 = W_130+ a *Ghimj[1074];
        W_131 = W_131+ a *Ghimj[1075];
        W_132 = W_132+ a *Ghimj[1076];
        W_133 = W_133+ a *Ghimj[1077];
        W_134 = W_134+ a *Ghimj[1078];
        W_135 = W_135+ a *Ghimj[1079];
        W_136 = W_136+ a *Ghimj[1080];
        W_137 = W_137+ a *Ghimj[1081];
        W_138 = W_138+ a *Ghimj[1082];
        a = - W_128/ Ghimj[1138];
        W_128 = -a;
        W_129 = W_129+ a *Ghimj[1139];
        W_130 = W_130+ a *Ghimj[1140];
        W_131 = W_131+ a *Ghimj[1141];
        W_132 = W_132+ a *Ghimj[1142];
        W_133 = W_133+ a *Ghimj[1143];
        W_134 = W_134+ a *Ghimj[1144];
        W_135 = W_135+ a *Ghimj[1145];
        W_136 = W_136+ a *Ghimj[1146];
        W_137 = W_137+ a *Ghimj[1147];
        W_138 = W_138+ a *Ghimj[1148];
        a = - W_129/ Ghimj[1176];
        W_129 = -a;
        W_130 = W_130+ a *Ghimj[1177];
        W_131 = W_131+ a *Ghimj[1178];
        W_132 = W_132+ a *Ghimj[1179];
        W_133 = W_133+ a *Ghimj[1180];
        W_134 = W_134+ a *Ghimj[1181];
        W_135 = W_135+ a *Ghimj[1182];
        W_136 = W_136+ a *Ghimj[1183];
        W_137 = W_137+ a *Ghimj[1184];
        W_138 = W_138+ a *Ghimj[1185];
        a = - W_130/ Ghimj[1218];
        W_130 = -a;
        W_131 = W_131+ a *Ghimj[1219];
        W_132 = W_132+ a *Ghimj[1220];
        W_133 = W_133+ a *Ghimj[1221];
        W_134 = W_134+ a *Ghimj[1222];
        W_135 = W_135+ a *Ghimj[1223];
        W_136 = W_136+ a *Ghimj[1224];
        W_137 = W_137+ a *Ghimj[1225];
        W_138 = W_138+ a *Ghimj[1226];
        a = - W_131/ Ghimj[1242];
        W_131 = -a;
        W_132 = W_132+ a *Ghimj[1243];
        W_133 = W_133+ a *Ghimj[1244];
        W_134 = W_134+ a *Ghimj[1245];
        W_135 = W_135+ a *Ghimj[1246];
        W_136 = W_136+ a *Ghimj[1247];
        W_137 = W_137+ a *Ghimj[1248];
        W_138 = W_138+ a *Ghimj[1249];
        a = - W_132/ Ghimj[1262];
        W_132 = -a;
        W_133 = W_133+ a *Ghimj[1263];
        W_134 = W_134+ a *Ghimj[1264];
        W_135 = W_135+ a *Ghimj[1265];
        W_136 = W_136+ a *Ghimj[1266];
        W_137 = W_137+ a *Ghimj[1267];
        W_138 = W_138+ a *Ghimj[1268];
        Ghimj[1269] = W_59;
        Ghimj[1270] = W_60;
        Ghimj[1271] = W_70;
        Ghimj[1272] = W_76;
        Ghimj[1273] = W_84;
        Ghimj[1274] = W_87;
        Ghimj[1275] = W_92;
        Ghimj[1276] = W_93;
        Ghimj[1277] = W_94;
        Ghimj[1278] = W_99;
        Ghimj[1279] = W_102;
        Ghimj[1280] = W_109;
        Ghimj[1281] = W_111;
        Ghimj[1282] = W_113;
        Ghimj[1283] = W_115;
        Ghimj[1284] = W_117;
        Ghimj[1285] = W_120;
        Ghimj[1286] = W_121;
        Ghimj[1287] = W_122;
        Ghimj[1288] = W_124;
        Ghimj[1289] = W_125;
        Ghimj[1290] = W_126;
        Ghimj[1291] = W_127;
        Ghimj[1292] = W_128;
        Ghimj[1293] = W_129;
        Ghimj[1294] = W_130;
        Ghimj[1295] = W_131;
        Ghimj[1296] = W_132;
        Ghimj[1297] = W_133;
        Ghimj[1298] = W_134;
        Ghimj[1299] = W_135;
        Ghimj[1300] = W_136;
        Ghimj[1301] = W_137;
        Ghimj[1302] = W_138;
        W_39 = Ghimj[1303];
        W_41 = Ghimj[1304];
        W_42 = Ghimj[1305];
        W_43 = Ghimj[1306];
        W_51 = Ghimj[1307];
        W_75 = Ghimj[1308];
        W_112 = Ghimj[1309];
        W_116 = Ghimj[1310];
        W_120 = Ghimj[1311];
        W_122 = Ghimj[1312];
        W_123 = Ghimj[1313];
        W_124 = Ghimj[1314];
        W_125 = Ghimj[1315];
        W_126 = Ghimj[1316];
        W_127 = Ghimj[1317];
        W_128 = Ghimj[1318];
        W_129 = Ghimj[1319];
        W_130 = Ghimj[1320];
        W_131 = Ghimj[1321];
        W_132 = Ghimj[1322];
        W_133 = Ghimj[1323];
        W_134 = Ghimj[1324];
        W_135 = Ghimj[1325];
        W_136 = Ghimj[1326];
        W_137 = Ghimj[1327];
        W_138 = Ghimj[1328];
        a = - W_39/ Ghimj[258];
        W_39 = -a;
        W_134 = W_134+ a *Ghimj[259];
        a = - W_41/ Ghimj[262];
        W_41 = -a;
        W_120 = W_120+ a *Ghimj[263];
        a = - W_42/ Ghimj[264];
        W_42 = -a;
        W_120 = W_120+ a *Ghimj[265];
        a = - W_43/ Ghimj[266];
        W_43 = -a;
        W_120 = W_120+ a *Ghimj[267];
        a = - W_51/ Ghimj[285];
        W_51 = -a;
        W_132 = W_132+ a *Ghimj[286];
        W_134 = W_134+ a *Ghimj[287];
        a = - W_75/ Ghimj[374];
        W_75 = -a;
        W_120 = W_120+ a *Ghimj[375];
        W_126 = W_126+ a *Ghimj[376];
        a = - W_112/ Ghimj[677];
        W_112 = -a;
        W_116 = W_116+ a *Ghimj[678];
        W_123 = W_123+ a *Ghimj[679];
        W_126 = W_126+ a *Ghimj[680];
        W_128 = W_128+ a *Ghimj[681];
        W_134 = W_134+ a *Ghimj[682];
        W_137 = W_137+ a *Ghimj[683];
        W_138 = W_138+ a *Ghimj[684];
        a = - W_116/ Ghimj[714];
        W_116 = -a;
        W_123 = W_123+ a *Ghimj[715];
        W_127 = W_127+ a *Ghimj[716];
        W_128 = W_128+ a *Ghimj[717];
        W_131 = W_131+ a *Ghimj[718];
        W_134 = W_134+ a *Ghimj[719];
        W_135 = W_135+ a *Ghimj[720];
        W_138 = W_138+ a *Ghimj[721];
        a = - W_120/ Ghimj[787];
        W_120 = -a;
        W_122 = W_122+ a *Ghimj[788];
        W_124 = W_124+ a *Ghimj[789];
        W_126 = W_126+ a *Ghimj[790];
        W_127 = W_127+ a *Ghimj[791];
        W_128 = W_128+ a *Ghimj[792];
        W_130 = W_130+ a *Ghimj[793];
        W_133 = W_133+ a *Ghimj[794];
        W_135 = W_135+ a *Ghimj[795];
        W_136 = W_136+ a *Ghimj[796];
        W_137 = W_137+ a *Ghimj[797];
        a = - W_122/ Ghimj[847];
        W_122 = -a;
        W_124 = W_124+ a *Ghimj[848];
        W_125 = W_125+ a *Ghimj[849];
        W_126 = W_126+ a *Ghimj[850];
        W_127 = W_127+ a *Ghimj[851];
        W_128 = W_128+ a *Ghimj[852];
        W_129 = W_129+ a *Ghimj[853];
        W_130 = W_130+ a *Ghimj[854];
        W_131 = W_131+ a *Ghimj[855];
        W_133 = W_133+ a *Ghimj[856];
        W_135 = W_135+ a *Ghimj[857];
        W_136 = W_136+ a *Ghimj[858];
        W_137 = W_137+ a *Ghimj[859];
        W_138 = W_138+ a *Ghimj[860];
        a = - W_123/ Ghimj[869];
        W_123 = -a;
        W_124 = W_124+ a *Ghimj[870];
        W_125 = W_125+ a *Ghimj[871];
        W_126 = W_126+ a *Ghimj[872];
        W_127 = W_127+ a *Ghimj[873];
        W_128 = W_128+ a *Ghimj[874];
        W_129 = W_129+ a *Ghimj[875];
        W_130 = W_130+ a *Ghimj[876];
        W_131 = W_131+ a *Ghimj[877];
        W_132 = W_132+ a *Ghimj[878];
        W_133 = W_133+ a *Ghimj[879];
        W_134 = W_134+ a *Ghimj[880];
        W_135 = W_135+ a *Ghimj[881];
        W_136 = W_136+ a *Ghimj[882];
        W_137 = W_137+ a *Ghimj[883];
        W_138 = W_138+ a *Ghimj[884];
        a = - W_124/ Ghimj[896];
        W_124 = -a;
        W_125 = W_125+ a *Ghimj[897];
        W_126 = W_126+ a *Ghimj[898];
        W_127 = W_127+ a *Ghimj[899];
        W_128 = W_128+ a *Ghimj[900];
        W_129 = W_129+ a *Ghimj[901];
        W_130 = W_130+ a *Ghimj[902];
        W_131 = W_131+ a *Ghimj[903];
        W_132 = W_132+ a *Ghimj[904];
        W_133 = W_133+ a *Ghimj[905];
        W_135 = W_135+ a *Ghimj[906];
        W_136 = W_136+ a *Ghimj[907];
        W_137 = W_137+ a *Ghimj[908];
        W_138 = W_138+ a *Ghimj[909];
        a = - W_125/ Ghimj[934];
        W_125 = -a;
        W_126 = W_126+ a *Ghimj[935];
        W_127 = W_127+ a *Ghimj[936];
        W_128 = W_128+ a *Ghimj[937];
        W_129 = W_129+ a *Ghimj[938];
        W_130 = W_130+ a *Ghimj[939];
        W_131 = W_131+ a *Ghimj[940];
        W_132 = W_132+ a *Ghimj[941];
        W_133 = W_133+ a *Ghimj[942];
        W_134 = W_134+ a *Ghimj[943];
        W_135 = W_135+ a *Ghimj[944];
        W_136 = W_136+ a *Ghimj[945];
        W_137 = W_137+ a *Ghimj[946];
        W_138 = W_138+ a *Ghimj[947];
        a = - W_126/ Ghimj[1023];
        W_126 = -a;
        W_127 = W_127+ a *Ghimj[1024];
        W_128 = W_128+ a *Ghimj[1025];
        W_129 = W_129+ a *Ghimj[1026];
        W_130 = W_130+ a *Ghimj[1027];
        W_131 = W_131+ a *Ghimj[1028];
        W_132 = W_132+ a *Ghimj[1029];
        W_133 = W_133+ a *Ghimj[1030];
        W_134 = W_134+ a *Ghimj[1031];
        W_135 = W_135+ a *Ghimj[1032];
        W_136 = W_136+ a *Ghimj[1033];
        W_137 = W_137+ a *Ghimj[1034];
        W_138 = W_138+ a *Ghimj[1035];
        a = - W_127/ Ghimj[1071];
        W_127 = -a;
        W_128 = W_128+ a *Ghimj[1072];
        W_129 = W_129+ a *Ghimj[1073];
        W_130 = W_130+ a *Ghimj[1074];
        W_131 = W_131+ a *Ghimj[1075];
        W_132 = W_132+ a *Ghimj[1076];
        W_133 = W_133+ a *Ghimj[1077];
        W_134 = W_134+ a *Ghimj[1078];
        W_135 = W_135+ a *Ghimj[1079];
        W_136 = W_136+ a *Ghimj[1080];
        W_137 = W_137+ a *Ghimj[1081];
        W_138 = W_138+ a *Ghimj[1082];
        a = - W_128/ Ghimj[1138];
        W_128 = -a;
        W_129 = W_129+ a *Ghimj[1139];
        W_130 = W_130+ a *Ghimj[1140];
        W_131 = W_131+ a *Ghimj[1141];
        W_132 = W_132+ a *Ghimj[1142];
        W_133 = W_133+ a *Ghimj[1143];
        W_134 = W_134+ a *Ghimj[1144];
        W_135 = W_135+ a *Ghimj[1145];
        W_136 = W_136+ a *Ghimj[1146];
        W_137 = W_137+ a *Ghimj[1147];
        W_138 = W_138+ a *Ghimj[1148];
        a = - W_129/ Ghimj[1176];
        W_129 = -a;
        W_130 = W_130+ a *Ghimj[1177];
        W_131 = W_131+ a *Ghimj[1178];
        W_132 = W_132+ a *Ghimj[1179];
        W_133 = W_133+ a *Ghimj[1180];
        W_134 = W_134+ a *Ghimj[1181];
        W_135 = W_135+ a *Ghimj[1182];
        W_136 = W_136+ a *Ghimj[1183];
        W_137 = W_137+ a *Ghimj[1184];
        W_138 = W_138+ a *Ghimj[1185];
        a = - W_130/ Ghimj[1218];
        W_130 = -a;
        W_131 = W_131+ a *Ghimj[1219];
        W_132 = W_132+ a *Ghimj[1220];
        W_133 = W_133+ a *Ghimj[1221];
        W_134 = W_134+ a *Ghimj[1222];
        W_135 = W_135+ a *Ghimj[1223];
        W_136 = W_136+ a *Ghimj[1224];
        W_137 = W_137+ a *Ghimj[1225];
        W_138 = W_138+ a *Ghimj[1226];
        a = - W_131/ Ghimj[1242];
        W_131 = -a;
        W_132 = W_132+ a *Ghimj[1243];
        W_133 = W_133+ a *Ghimj[1244];
        W_134 = W_134+ a *Ghimj[1245];
        W_135 = W_135+ a *Ghimj[1246];
        W_136 = W_136+ a *Ghimj[1247];
        W_137 = W_137+ a *Ghimj[1248];
        W_138 = W_138+ a *Ghimj[1249];
        a = - W_132/ Ghimj[1262];
        W_132 = -a;
        W_133 = W_133+ a *Ghimj[1263];
        W_134 = W_134+ a *Ghimj[1264];
        W_135 = W_135+ a *Ghimj[1265];
        W_136 = W_136+ a *Ghimj[1266];
        W_137 = W_137+ a *Ghimj[1267];
        W_138 = W_138+ a *Ghimj[1268];
        a = - W_133/ Ghimj[1297];
        W_133 = -a;
        W_134 = W_134+ a *Ghimj[1298];
        W_135 = W_135+ a *Ghimj[1299];
        W_136 = W_136+ a *Ghimj[1300];
        W_137 = W_137+ a *Ghimj[1301];
        W_138 = W_138+ a *Ghimj[1302];
        Ghimj[1303] = W_39;
        Ghimj[1304] = W_41;
        Ghimj[1305] = W_42;
        Ghimj[1306] = W_43;
        Ghimj[1307] = W_51;
        Ghimj[1308] = W_75;
        Ghimj[1309] = W_112;
        Ghimj[1310] = W_116;
        Ghimj[1311] = W_120;
        Ghimj[1312] = W_122;
        Ghimj[1313] = W_123;
        Ghimj[1314] = W_124;
        Ghimj[1315] = W_125;
        Ghimj[1316] = W_126;
        Ghimj[1317] = W_127;
        Ghimj[1318] = W_128;
        Ghimj[1319] = W_129;
        Ghimj[1320] = W_130;
        Ghimj[1321] = W_131;
        Ghimj[1322] = W_132;
        Ghimj[1323] = W_133;
        Ghimj[1324] = W_134;
        Ghimj[1325] = W_135;
        Ghimj[1326] = W_136;
        Ghimj[1327] = W_137;
        Ghimj[1328] = W_138;
        W_0 = Ghimj[1329];
        W_50 = Ghimj[1330];
        W_58 = Ghimj[1331];
        W_59 = Ghimj[1332];
        W_62 = Ghimj[1333];
        W_64 = Ghimj[1334];
        W_73 = Ghimj[1335];
        W_76 = Ghimj[1336];
        W_77 = Ghimj[1337];
        W_83 = Ghimj[1338];
        W_87 = Ghimj[1339];
        W_91 = Ghimj[1340];
        W_92 = Ghimj[1341];
        W_93 = Ghimj[1342];
        W_94 = Ghimj[1343];
        W_99 = Ghimj[1344];
        W_101 = Ghimj[1345];
        W_102 = Ghimj[1346];
        W_105 = Ghimj[1347];
        W_106 = Ghimj[1348];
        W_109 = Ghimj[1349];
        W_111 = Ghimj[1350];
        W_113 = Ghimj[1351];
        W_114 = Ghimj[1352];
        W_115 = Ghimj[1353];
        W_116 = Ghimj[1354];
        W_117 = Ghimj[1355];
        W_119 = Ghimj[1356];
        W_121 = Ghimj[1357];
        W_123 = Ghimj[1358];
        W_124 = Ghimj[1359];
        W_125 = Ghimj[1360];
        W_126 = Ghimj[1361];
        W_127 = Ghimj[1362];
        W_128 = Ghimj[1363];
        W_129 = Ghimj[1364];
        W_130 = Ghimj[1365];
        W_131 = Ghimj[1366];
        W_132 = Ghimj[1367];
        W_133 = Ghimj[1368];
        W_134 = Ghimj[1369];
        W_135 = Ghimj[1370];
        W_136 = Ghimj[1371];
        W_137 = Ghimj[1372];
        W_138 = Ghimj[1373];
        a = - W_0/ Ghimj[0];
        W_0 = -a;
        a = - W_50/ Ghimj[282];
        W_50 = -a;
        W_83 = W_83+ a *Ghimj[283];
        W_138 = W_138+ a *Ghimj[284];
        a = - W_58/ Ghimj[303];
        W_58 = -a;
        W_91 = W_91+ a *Ghimj[304];
        W_126 = W_126+ a *Ghimj[305];
        a = - W_59/ Ghimj[306];
        W_59 = -a;
        W_133 = W_133+ a *Ghimj[307];
        W_135 = W_135+ a *Ghimj[308];
        a = - W_62/ Ghimj[319];
        W_62 = -a;
        W_93 = W_93+ a *Ghimj[320];
        W_126 = W_126+ a *Ghimj[321];
        W_133 = W_133+ a *Ghimj[322];
        a = - W_64/ Ghimj[327];
        W_64 = -a;
        W_113 = W_113+ a *Ghimj[328];
        W_126 = W_126+ a *Ghimj[329];
        W_135 = W_135+ a *Ghimj[330];
        a = - W_73/ Ghimj[364];
        W_73 = -a;
        W_126 = W_126+ a *Ghimj[365];
        W_135 = W_135+ a *Ghimj[366];
        W_137 = W_137+ a *Ghimj[367];
        a = - W_76/ Ghimj[377];
        W_76 = -a;
        W_87 = W_87+ a *Ghimj[378];
        W_126 = W_126+ a *Ghimj[379];
        W_133 = W_133+ a *Ghimj[380];
        W_135 = W_135+ a *Ghimj[381];
        a = - W_77/ Ghimj[382];
        W_77 = -a;
        W_121 = W_121+ a *Ghimj[383];
        W_126 = W_126+ a *Ghimj[384];
        W_135 = W_135+ a *Ghimj[385];
        a = - W_83/ Ghimj[416];
        W_83 = -a;
        W_128 = W_128+ a *Ghimj[417];
        W_135 = W_135+ a *Ghimj[418];
        W_136 = W_136+ a *Ghimj[419];
        W_138 = W_138+ a *Ghimj[420];
        a = - W_87/ Ghimj[444];
        W_87 = -a;
        W_92 = W_92+ a *Ghimj[445];
        W_124 = W_124+ a *Ghimj[446];
        W_126 = W_126+ a *Ghimj[447];
        W_135 = W_135+ a *Ghimj[448];
        W_137 = W_137+ a *Ghimj[449];
        a = - W_91/ Ghimj[481];
        W_91 = -a;
        W_106 = W_106+ a *Ghimj[482];
        W_109 = W_109+ a *Ghimj[483];
        W_126 = W_126+ a *Ghimj[484];
        W_133 = W_133+ a *Ghimj[485];
        W_136 = W_136+ a *Ghimj[486];
        a = - W_92/ Ghimj[489];
        W_92 = -a;
        W_124 = W_124+ a *Ghimj[490];
        W_126 = W_126+ a *Ghimj[491];
        W_133 = W_133+ a *Ghimj[492];
        W_135 = W_135+ a *Ghimj[493];
        W_137 = W_137+ a *Ghimj[494];
        a = - W_93/ Ghimj[497];
        W_93 = -a;
        W_125 = W_125+ a *Ghimj[498];
        W_126 = W_126+ a *Ghimj[499];
        W_133 = W_133+ a *Ghimj[500];
        W_137 = W_137+ a *Ghimj[501];
        a = - W_94/ Ghimj[505];
        W_94 = -a;
        W_125 = W_125+ a *Ghimj[506];
        W_126 = W_126+ a *Ghimj[507];
        W_133 = W_133+ a *Ghimj[508];
        W_137 = W_137+ a *Ghimj[509];
        a = - W_99/ Ghimj[565];
        W_99 = -a;
        W_102 = W_102+ a *Ghimj[566];
        W_111 = W_111+ a *Ghimj[567];
        W_125 = W_125+ a *Ghimj[568];
        W_126 = W_126+ a *Ghimj[569];
        W_133 = W_133+ a *Ghimj[570];
        W_137 = W_137+ a *Ghimj[571];
        a = - W_101/ Ghimj[586];
        W_101 = -a;
        W_105 = W_105+ a *Ghimj[587];
        W_114 = W_114+ a *Ghimj[588];
        W_116 = W_116+ a *Ghimj[589];
        W_119 = W_119+ a *Ghimj[590];
        W_123 = W_123+ a *Ghimj[591];
        W_126 = W_126+ a *Ghimj[592];
        W_128 = W_128+ a *Ghimj[593];
        W_130 = W_130+ a *Ghimj[594];
        W_135 = W_135+ a *Ghimj[595];
        W_136 = W_136+ a *Ghimj[596];
        W_138 = W_138+ a *Ghimj[597];
        a = - W_102/ Ghimj[600];
        W_102 = -a;
        W_125 = W_125+ a *Ghimj[601];
        W_126 = W_126+ a *Ghimj[602];
        W_133 = W_133+ a *Ghimj[603];
        W_137 = W_137+ a *Ghimj[604];
        a = - W_105/ Ghimj[616];
        W_105 = -a;
        W_128 = W_128+ a *Ghimj[617];
        W_129 = W_129+ a *Ghimj[618];
        W_132 = W_132+ a *Ghimj[619];
        W_135 = W_135+ a *Ghimj[620];
        W_138 = W_138+ a *Ghimj[621];
        a = - W_106/ Ghimj[622];
        W_106 = -a;
        W_124 = W_124+ a *Ghimj[623];
        W_126 = W_126+ a *Ghimj[624];
        W_136 = W_136+ a *Ghimj[625];
        a = - W_109/ Ghimj[648];
        W_109 = -a;
        W_124 = W_124+ a *Ghimj[649];
        W_125 = W_125+ a *Ghimj[650];
        W_126 = W_126+ a *Ghimj[651];
        W_133 = W_133+ a *Ghimj[652];
        W_136 = W_136+ a *Ghimj[653];
        W_137 = W_137+ a *Ghimj[654];
        a = - W_111/ Ghimj[669];
        W_111 = -a;
        W_115 = W_115+ a *Ghimj[670];
        W_124 = W_124+ a *Ghimj[671];
        W_125 = W_125+ a *Ghimj[672];
        W_126 = W_126+ a *Ghimj[673];
        W_133 = W_133+ a *Ghimj[674];
        W_136 = W_136+ a *Ghimj[675];
        W_137 = W_137+ a *Ghimj[676];
        a = - W_113/ Ghimj[689];
        W_113 = -a;
        W_124 = W_124+ a *Ghimj[690];
        W_125 = W_125+ a *Ghimj[691];
        W_126 = W_126+ a *Ghimj[692];
        W_133 = W_133+ a *Ghimj[693];
        W_135 = W_135+ a *Ghimj[694];
        W_136 = W_136+ a *Ghimj[695];
        W_137 = W_137+ a *Ghimj[696];
        a = - W_114/ Ghimj[697];
        W_114 = -a;
        W_126 = W_126+ a *Ghimj[698];
        W_127 = W_127+ a *Ghimj[699];
        W_129 = W_129+ a *Ghimj[700];
        W_132 = W_132+ a *Ghimj[701];
        W_136 = W_136+ a *Ghimj[702];
        a = - W_115/ Ghimj[706];
        W_115 = -a;
        W_124 = W_124+ a *Ghimj[707];
        W_126 = W_126+ a *Ghimj[708];
        W_127 = W_127+ a *Ghimj[709];
        W_129 = W_129+ a *Ghimj[710];
        W_133 = W_133+ a *Ghimj[711];
        W_136 = W_136+ a *Ghimj[712];
        W_137 = W_137+ a *Ghimj[713];
        a = - W_116/ Ghimj[714];
        W_116 = -a;
        W_123 = W_123+ a *Ghimj[715];
        W_127 = W_127+ a *Ghimj[716];
        W_128 = W_128+ a *Ghimj[717];
        W_131 = W_131+ a *Ghimj[718];
        W_134 = W_134+ a *Ghimj[719];
        W_135 = W_135+ a *Ghimj[720];
        W_138 = W_138+ a *Ghimj[721];
        a = - W_117/ Ghimj[731];
        W_117 = -a;
        W_121 = W_121+ a *Ghimj[732];
        W_124 = W_124+ a *Ghimj[733];
        W_125 = W_125+ a *Ghimj[734];
        W_126 = W_126+ a *Ghimj[735];
        W_127 = W_127+ a *Ghimj[736];
        W_129 = W_129+ a *Ghimj[737];
        W_133 = W_133+ a *Ghimj[738];
        W_136 = W_136+ a *Ghimj[739];
        W_137 = W_137+ a *Ghimj[740];
        a = - W_119/ Ghimj[767];
        W_119 = -a;
        W_121 = W_121+ a *Ghimj[768];
        W_124 = W_124+ a *Ghimj[769];
        W_125 = W_125+ a *Ghimj[770];
        W_126 = W_126+ a *Ghimj[771];
        W_127 = W_127+ a *Ghimj[772];
        W_129 = W_129+ a *Ghimj[773];
        W_133 = W_133+ a *Ghimj[774];
        W_136 = W_136+ a *Ghimj[775];
        W_137 = W_137+ a *Ghimj[776];
        a = - W_121/ Ghimj[821];
        W_121 = -a;
        W_124 = W_124+ a *Ghimj[822];
        W_125 = W_125+ a *Ghimj[823];
        W_126 = W_126+ a *Ghimj[824];
        W_127 = W_127+ a *Ghimj[825];
        W_129 = W_129+ a *Ghimj[826];
        W_133 = W_133+ a *Ghimj[827];
        W_135 = W_135+ a *Ghimj[828];
        W_136 = W_136+ a *Ghimj[829];
        W_137 = W_137+ a *Ghimj[830];
        a = - W_123/ Ghimj[869];
        W_123 = -a;
        W_124 = W_124+ a *Ghimj[870];
        W_125 = W_125+ a *Ghimj[871];
        W_126 = W_126+ a *Ghimj[872];
        W_127 = W_127+ a *Ghimj[873];
        W_128 = W_128+ a *Ghimj[874];
        W_129 = W_129+ a *Ghimj[875];
        W_130 = W_130+ a *Ghimj[876];
        W_131 = W_131+ a *Ghimj[877];
        W_132 = W_132+ a *Ghimj[878];
        W_133 = W_133+ a *Ghimj[879];
        W_134 = W_134+ a *Ghimj[880];
        W_135 = W_135+ a *Ghimj[881];
        W_136 = W_136+ a *Ghimj[882];
        W_137 = W_137+ a *Ghimj[883];
        W_138 = W_138+ a *Ghimj[884];
        a = - W_124/ Ghimj[896];
        W_124 = -a;
        W_125 = W_125+ a *Ghimj[897];
        W_126 = W_126+ a *Ghimj[898];
        W_127 = W_127+ a *Ghimj[899];
        W_128 = W_128+ a *Ghimj[900];
        W_129 = W_129+ a *Ghimj[901];
        W_130 = W_130+ a *Ghimj[902];
        W_131 = W_131+ a *Ghimj[903];
        W_132 = W_132+ a *Ghimj[904];
        W_133 = W_133+ a *Ghimj[905];
        W_135 = W_135+ a *Ghimj[906];
        W_136 = W_136+ a *Ghimj[907];
        W_137 = W_137+ a *Ghimj[908];
        W_138 = W_138+ a *Ghimj[909];
        a = - W_125/ Ghimj[934];
        W_125 = -a;
        W_126 = W_126+ a *Ghimj[935];
        W_127 = W_127+ a *Ghimj[936];
        W_128 = W_128+ a *Ghimj[937];
        W_129 = W_129+ a *Ghimj[938];
        W_130 = W_130+ a *Ghimj[939];
        W_131 = W_131+ a *Ghimj[940];
        W_132 = W_132+ a *Ghimj[941];
        W_133 = W_133+ a *Ghimj[942];
        W_134 = W_134+ a *Ghimj[943];
        W_135 = W_135+ a *Ghimj[944];
        W_136 = W_136+ a *Ghimj[945];
        W_137 = W_137+ a *Ghimj[946];
        W_138 = W_138+ a *Ghimj[947];
        a = - W_126/ Ghimj[1023];
        W_126 = -a;
        W_127 = W_127+ a *Ghimj[1024];
        W_128 = W_128+ a *Ghimj[1025];
        W_129 = W_129+ a *Ghimj[1026];
        W_130 = W_130+ a *Ghimj[1027];
        W_131 = W_131+ a *Ghimj[1028];
        W_132 = W_132+ a *Ghimj[1029];
        W_133 = W_133+ a *Ghimj[1030];
        W_134 = W_134+ a *Ghimj[1031];
        W_135 = W_135+ a *Ghimj[1032];
        W_136 = W_136+ a *Ghimj[1033];
        W_137 = W_137+ a *Ghimj[1034];
        W_138 = W_138+ a *Ghimj[1035];
        a = - W_127/ Ghimj[1071];
        W_127 = -a;
        W_128 = W_128+ a *Ghimj[1072];
        W_129 = W_129+ a *Ghimj[1073];
        W_130 = W_130+ a *Ghimj[1074];
        W_131 = W_131+ a *Ghimj[1075];
        W_132 = W_132+ a *Ghimj[1076];
        W_133 = W_133+ a *Ghimj[1077];
        W_134 = W_134+ a *Ghimj[1078];
        W_135 = W_135+ a *Ghimj[1079];
        W_136 = W_136+ a *Ghimj[1080];
        W_137 = W_137+ a *Ghimj[1081];
        W_138 = W_138+ a *Ghimj[1082];
        a = - W_128/ Ghimj[1138];
        W_128 = -a;
        W_129 = W_129+ a *Ghimj[1139];
        W_130 = W_130+ a *Ghimj[1140];
        W_131 = W_131+ a *Ghimj[1141];
        W_132 = W_132+ a *Ghimj[1142];
        W_133 = W_133+ a *Ghimj[1143];
        W_134 = W_134+ a *Ghimj[1144];
        W_135 = W_135+ a *Ghimj[1145];
        W_136 = W_136+ a *Ghimj[1146];
        W_137 = W_137+ a *Ghimj[1147];
        W_138 = W_138+ a *Ghimj[1148];
        a = - W_129/ Ghimj[1176];
        W_129 = -a;
        W_130 = W_130+ a *Ghimj[1177];
        W_131 = W_131+ a *Ghimj[1178];
        W_132 = W_132+ a *Ghimj[1179];
        W_133 = W_133+ a *Ghimj[1180];
        W_134 = W_134+ a *Ghimj[1181];
        W_135 = W_135+ a *Ghimj[1182];
        W_136 = W_136+ a *Ghimj[1183];
        W_137 = W_137+ a *Ghimj[1184];
        W_138 = W_138+ a *Ghimj[1185];
        a = - W_130/ Ghimj[1218];
        W_130 = -a;
        W_131 = W_131+ a *Ghimj[1219];
        W_132 = W_132+ a *Ghimj[1220];
        W_133 = W_133+ a *Ghimj[1221];
        W_134 = W_134+ a *Ghimj[1222];
        W_135 = W_135+ a *Ghimj[1223];
        W_136 = W_136+ a *Ghimj[1224];
        W_137 = W_137+ a *Ghimj[1225];
        W_138 = W_138+ a *Ghimj[1226];
        a = - W_131/ Ghimj[1242];
        W_131 = -a;
        W_132 = W_132+ a *Ghimj[1243];
        W_133 = W_133+ a *Ghimj[1244];
        W_134 = W_134+ a *Ghimj[1245];
        W_135 = W_135+ a *Ghimj[1246];
        W_136 = W_136+ a *Ghimj[1247];
        W_137 = W_137+ a *Ghimj[1248];
        W_138 = W_138+ a *Ghimj[1249];
        a = - W_132/ Ghimj[1262];
        W_132 = -a;
        W_133 = W_133+ a *Ghimj[1263];
        W_134 = W_134+ a *Ghimj[1264];
        W_135 = W_135+ a *Ghimj[1265];
        W_136 = W_136+ a *Ghimj[1266];
        W_137 = W_137+ a *Ghimj[1267];
        W_138 = W_138+ a *Ghimj[1268];
        a = - W_133/ Ghimj[1297];
        W_133 = -a;
        W_134 = W_134+ a *Ghimj[1298];
        W_135 = W_135+ a *Ghimj[1299];
        W_136 = W_136+ a *Ghimj[1300];
        W_137 = W_137+ a *Ghimj[1301];
        W_138 = W_138+ a *Ghimj[1302];
        a = - W_134/ Ghimj[1324];
        W_134 = -a;
        W_135 = W_135+ a *Ghimj[1325];
        W_136 = W_136+ a *Ghimj[1326];
        W_137 = W_137+ a *Ghimj[1327];
        W_138 = W_138+ a *Ghimj[1328];
        Ghimj[1329] = W_0;
        Ghimj[1330] = W_50;
        Ghimj[1331] = W_58;
        Ghimj[1332] = W_59;
        Ghimj[1333] = W_62;
        Ghimj[1334] = W_64;
        Ghimj[1335] = W_73;
        Ghimj[1336] = W_76;
        Ghimj[1337] = W_77;
        Ghimj[1338] = W_83;
        Ghimj[1339] = W_87;
        Ghimj[1340] = W_91;
        Ghimj[1341] = W_92;
        Ghimj[1342] = W_93;
        Ghimj[1343] = W_94;
        Ghimj[1344] = W_99;
        Ghimj[1345] = W_101;
        Ghimj[1346] = W_102;
        Ghimj[1347] = W_105;
        Ghimj[1348] = W_106;
        Ghimj[1349] = W_109;
        Ghimj[1350] = W_111;
        Ghimj[1351] = W_113;
        Ghimj[1352] = W_114;
        Ghimj[1353] = W_115;
        Ghimj[1354] = W_116;
        Ghimj[1355] = W_117;
        Ghimj[1356] = W_119;
        Ghimj[1357] = W_121;
        Ghimj[1358] = W_123;
        Ghimj[1359] = W_124;
        Ghimj[1360] = W_125;
        Ghimj[1361] = W_126;
        Ghimj[1362] = W_127;
        Ghimj[1363] = W_128;
        Ghimj[1364] = W_129;
        Ghimj[1365] = W_130;
        Ghimj[1366] = W_131;
        Ghimj[1367] = W_132;
        Ghimj[1368] = W_133;
        Ghimj[1369] = W_134;
        Ghimj[1370] = W_135;
        Ghimj[1371] = W_136;
        Ghimj[1372] = W_137;
        Ghimj[1373] = W_138;
        W_73 = Ghimj[1374];
        W_83 = Ghimj[1375];
        W_101 = Ghimj[1376];
        W_105 = Ghimj[1377];
        W_106 = Ghimj[1378];
        W_107 = Ghimj[1379];
        W_114 = Ghimj[1380];
        W_116 = Ghimj[1381];
        W_117 = Ghimj[1382];
        W_119 = Ghimj[1383];
        W_121 = Ghimj[1384];
        W_123 = Ghimj[1385];
        W_124 = Ghimj[1386];
        W_125 = Ghimj[1387];
        W_126 = Ghimj[1388];
        W_127 = Ghimj[1389];
        W_128 = Ghimj[1390];
        W_129 = Ghimj[1391];
        W_130 = Ghimj[1392];
        W_131 = Ghimj[1393];
        W_132 = Ghimj[1394];
        W_133 = Ghimj[1395];
        W_134 = Ghimj[1396];
        W_135 = Ghimj[1397];
        W_136 = Ghimj[1398];
        W_137 = Ghimj[1399];
        W_138 = Ghimj[1400];
        a = - W_73/ Ghimj[364];
        W_73 = -a;
        W_126 = W_126+ a *Ghimj[365];
        W_135 = W_135+ a *Ghimj[366];
        W_137 = W_137+ a *Ghimj[367];
        a = - W_83/ Ghimj[416];
        W_83 = -a;
        W_128 = W_128+ a *Ghimj[417];
        W_135 = W_135+ a *Ghimj[418];
        W_136 = W_136+ a *Ghimj[419];
        W_138 = W_138+ a *Ghimj[420];
        a = - W_101/ Ghimj[586];
        W_101 = -a;
        W_105 = W_105+ a *Ghimj[587];
        W_114 = W_114+ a *Ghimj[588];
        W_116 = W_116+ a *Ghimj[589];
        W_119 = W_119+ a *Ghimj[590];
        W_123 = W_123+ a *Ghimj[591];
        W_126 = W_126+ a *Ghimj[592];
        W_128 = W_128+ a *Ghimj[593];
        W_130 = W_130+ a *Ghimj[594];
        W_135 = W_135+ a *Ghimj[595];
        W_136 = W_136+ a *Ghimj[596];
        W_138 = W_138+ a *Ghimj[597];
        a = - W_105/ Ghimj[616];
        W_105 = -a;
        W_128 = W_128+ a *Ghimj[617];
        W_129 = W_129+ a *Ghimj[618];
        W_132 = W_132+ a *Ghimj[619];
        W_135 = W_135+ a *Ghimj[620];
        W_138 = W_138+ a *Ghimj[621];
        a = - W_106/ Ghimj[622];
        W_106 = -a;
        W_124 = W_124+ a *Ghimj[623];
        W_126 = W_126+ a *Ghimj[624];
        W_136 = W_136+ a *Ghimj[625];
        a = - W_107/ Ghimj[626];
        W_107 = -a;
        W_124 = W_124+ a *Ghimj[627];
        W_126 = W_126+ a *Ghimj[628];
        W_136 = W_136+ a *Ghimj[629];
        a = - W_114/ Ghimj[697];
        W_114 = -a;
        W_126 = W_126+ a *Ghimj[698];
        W_127 = W_127+ a *Ghimj[699];
        W_129 = W_129+ a *Ghimj[700];
        W_132 = W_132+ a *Ghimj[701];
        W_136 = W_136+ a *Ghimj[702];
        a = - W_116/ Ghimj[714];
        W_116 = -a;
        W_123 = W_123+ a *Ghimj[715];
        W_127 = W_127+ a *Ghimj[716];
        W_128 = W_128+ a *Ghimj[717];
        W_131 = W_131+ a *Ghimj[718];
        W_134 = W_134+ a *Ghimj[719];
        W_135 = W_135+ a *Ghimj[720];
        W_138 = W_138+ a *Ghimj[721];
        a = - W_117/ Ghimj[731];
        W_117 = -a;
        W_121 = W_121+ a *Ghimj[732];
        W_124 = W_124+ a *Ghimj[733];
        W_125 = W_125+ a *Ghimj[734];
        W_126 = W_126+ a *Ghimj[735];
        W_127 = W_127+ a *Ghimj[736];
        W_129 = W_129+ a *Ghimj[737];
        W_133 = W_133+ a *Ghimj[738];
        W_136 = W_136+ a *Ghimj[739];
        W_137 = W_137+ a *Ghimj[740];
        a = - W_119/ Ghimj[767];
        W_119 = -a;
        W_121 = W_121+ a *Ghimj[768];
        W_124 = W_124+ a *Ghimj[769];
        W_125 = W_125+ a *Ghimj[770];
        W_126 = W_126+ a *Ghimj[771];
        W_127 = W_127+ a *Ghimj[772];
        W_129 = W_129+ a *Ghimj[773];
        W_133 = W_133+ a *Ghimj[774];
        W_136 = W_136+ a *Ghimj[775];
        W_137 = W_137+ a *Ghimj[776];
        a = - W_121/ Ghimj[821];
        W_121 = -a;
        W_124 = W_124+ a *Ghimj[822];
        W_125 = W_125+ a *Ghimj[823];
        W_126 = W_126+ a *Ghimj[824];
        W_127 = W_127+ a *Ghimj[825];
        W_129 = W_129+ a *Ghimj[826];
        W_133 = W_133+ a *Ghimj[827];
        W_135 = W_135+ a *Ghimj[828];
        W_136 = W_136+ a *Ghimj[829];
        W_137 = W_137+ a *Ghimj[830];
        a = - W_123/ Ghimj[869];
        W_123 = -a;
        W_124 = W_124+ a *Ghimj[870];
        W_125 = W_125+ a *Ghimj[871];
        W_126 = W_126+ a *Ghimj[872];
        W_127 = W_127+ a *Ghimj[873];
        W_128 = W_128+ a *Ghimj[874];
        W_129 = W_129+ a *Ghimj[875];
        W_130 = W_130+ a *Ghimj[876];
        W_131 = W_131+ a *Ghimj[877];
        W_132 = W_132+ a *Ghimj[878];
        W_133 = W_133+ a *Ghimj[879];
        W_134 = W_134+ a *Ghimj[880];
        W_135 = W_135+ a *Ghimj[881];
        W_136 = W_136+ a *Ghimj[882];
        W_137 = W_137+ a *Ghimj[883];
        W_138 = W_138+ a *Ghimj[884];
        a = - W_124/ Ghimj[896];
        W_124 = -a;
        W_125 = W_125+ a *Ghimj[897];
        W_126 = W_126+ a *Ghimj[898];
        W_127 = W_127+ a *Ghimj[899];
        W_128 = W_128+ a *Ghimj[900];
        W_129 = W_129+ a *Ghimj[901];
        W_130 = W_130+ a *Ghimj[902];
        W_131 = W_131+ a *Ghimj[903];
        W_132 = W_132+ a *Ghimj[904];
        W_133 = W_133+ a *Ghimj[905];
        W_135 = W_135+ a *Ghimj[906];
        W_136 = W_136+ a *Ghimj[907];
        W_137 = W_137+ a *Ghimj[908];
        W_138 = W_138+ a *Ghimj[909];
        a = - W_125/ Ghimj[934];
        W_125 = -a;
        W_126 = W_126+ a *Ghimj[935];
        W_127 = W_127+ a *Ghimj[936];
        W_128 = W_128+ a *Ghimj[937];
        W_129 = W_129+ a *Ghimj[938];
        W_130 = W_130+ a *Ghimj[939];
        W_131 = W_131+ a *Ghimj[940];
        W_132 = W_132+ a *Ghimj[941];
        W_133 = W_133+ a *Ghimj[942];
        W_134 = W_134+ a *Ghimj[943];
        W_135 = W_135+ a *Ghimj[944];
        W_136 = W_136+ a *Ghimj[945];
        W_137 = W_137+ a *Ghimj[946];
        W_138 = W_138+ a *Ghimj[947];
        a = - W_126/ Ghimj[1023];
        W_126 = -a;
        W_127 = W_127+ a *Ghimj[1024];
        W_128 = W_128+ a *Ghimj[1025];
        W_129 = W_129+ a *Ghimj[1026];
        W_130 = W_130+ a *Ghimj[1027];
        W_131 = W_131+ a *Ghimj[1028];
        W_132 = W_132+ a *Ghimj[1029];
        W_133 = W_133+ a *Ghimj[1030];
        W_134 = W_134+ a *Ghimj[1031];
        W_135 = W_135+ a *Ghimj[1032];
        W_136 = W_136+ a *Ghimj[1033];
        W_137 = W_137+ a *Ghimj[1034];
        W_138 = W_138+ a *Ghimj[1035];
        a = - W_127/ Ghimj[1071];
        W_127 = -a;
        W_128 = W_128+ a *Ghimj[1072];
        W_129 = W_129+ a *Ghimj[1073];
        W_130 = W_130+ a *Ghimj[1074];
        W_131 = W_131+ a *Ghimj[1075];
        W_132 = W_132+ a *Ghimj[1076];
        W_133 = W_133+ a *Ghimj[1077];
        W_134 = W_134+ a *Ghimj[1078];
        W_135 = W_135+ a *Ghimj[1079];
        W_136 = W_136+ a *Ghimj[1080];
        W_137 = W_137+ a *Ghimj[1081];
        W_138 = W_138+ a *Ghimj[1082];
        a = - W_128/ Ghimj[1138];
        W_128 = -a;
        W_129 = W_129+ a *Ghimj[1139];
        W_130 = W_130+ a *Ghimj[1140];
        W_131 = W_131+ a *Ghimj[1141];
        W_132 = W_132+ a *Ghimj[1142];
        W_133 = W_133+ a *Ghimj[1143];
        W_134 = W_134+ a *Ghimj[1144];
        W_135 = W_135+ a *Ghimj[1145];
        W_136 = W_136+ a *Ghimj[1146];
        W_137 = W_137+ a *Ghimj[1147];
        W_138 = W_138+ a *Ghimj[1148];
        a = - W_129/ Ghimj[1176];
        W_129 = -a;
        W_130 = W_130+ a *Ghimj[1177];
        W_131 = W_131+ a *Ghimj[1178];
        W_132 = W_132+ a *Ghimj[1179];
        W_133 = W_133+ a *Ghimj[1180];
        W_134 = W_134+ a *Ghimj[1181];
        W_135 = W_135+ a *Ghimj[1182];
        W_136 = W_136+ a *Ghimj[1183];
        W_137 = W_137+ a *Ghimj[1184];
        W_138 = W_138+ a *Ghimj[1185];
        a = - W_130/ Ghimj[1218];
        W_130 = -a;
        W_131 = W_131+ a *Ghimj[1219];
        W_132 = W_132+ a *Ghimj[1220];
        W_133 = W_133+ a *Ghimj[1221];
        W_134 = W_134+ a *Ghimj[1222];
        W_135 = W_135+ a *Ghimj[1223];
        W_136 = W_136+ a *Ghimj[1224];
        W_137 = W_137+ a *Ghimj[1225];
        W_138 = W_138+ a *Ghimj[1226];
        a = - W_131/ Ghimj[1242];
        W_131 = -a;
        W_132 = W_132+ a *Ghimj[1243];
        W_133 = W_133+ a *Ghimj[1244];
        W_134 = W_134+ a *Ghimj[1245];
        W_135 = W_135+ a *Ghimj[1246];
        W_136 = W_136+ a *Ghimj[1247];
        W_137 = W_137+ a *Ghimj[1248];
        W_138 = W_138+ a *Ghimj[1249];
        a = - W_132/ Ghimj[1262];
        W_132 = -a;
        W_133 = W_133+ a *Ghimj[1263];
        W_134 = W_134+ a *Ghimj[1264];
        W_135 = W_135+ a *Ghimj[1265];
        W_136 = W_136+ a *Ghimj[1266];
        W_137 = W_137+ a *Ghimj[1267];
        W_138 = W_138+ a *Ghimj[1268];
        a = - W_133/ Ghimj[1297];
        W_133 = -a;
        W_134 = W_134+ a *Ghimj[1298];
        W_135 = W_135+ a *Ghimj[1299];
        W_136 = W_136+ a *Ghimj[1300];
        W_137 = W_137+ a *Ghimj[1301];
        W_138 = W_138+ a *Ghimj[1302];
        a = - W_134/ Ghimj[1324];
        W_134 = -a;
        W_135 = W_135+ a *Ghimj[1325];
        W_136 = W_136+ a *Ghimj[1326];
        W_137 = W_137+ a *Ghimj[1327];
        W_138 = W_138+ a *Ghimj[1328];
        a = - W_135/ Ghimj[1370];
        W_135 = -a;
        W_136 = W_136+ a *Ghimj[1371];
        W_137 = W_137+ a *Ghimj[1372];
        W_138 = W_138+ a *Ghimj[1373];
        Ghimj[1374] = W_73;
        Ghimj[1375] = W_83;
        Ghimj[1376] = W_101;
        Ghimj[1377] = W_105;
        Ghimj[1378] = W_106;
        Ghimj[1379] = W_107;
        Ghimj[1380] = W_114;
        Ghimj[1381] = W_116;
        Ghimj[1382] = W_117;
        Ghimj[1383] = W_119;
        Ghimj[1384] = W_121;
        Ghimj[1385] = W_123;
        Ghimj[1386] = W_124;
        Ghimj[1387] = W_125;
        Ghimj[1388] = W_126;
        Ghimj[1389] = W_127;
        Ghimj[1390] = W_128;
        Ghimj[1391] = W_129;
        Ghimj[1392] = W_130;
        Ghimj[1393] = W_131;
        Ghimj[1394] = W_132;
        Ghimj[1395] = W_133;
        Ghimj[1396] = W_134;
        Ghimj[1397] = W_135;
        Ghimj[1398] = W_136;
        Ghimj[1399] = W_137;
        Ghimj[1400] = W_138;
        W_46 = Ghimj[1401];
        W_56 = Ghimj[1402];
        W_62 = Ghimj[1403];
        W_65 = Ghimj[1404];
        W_66 = Ghimj[1405];
        W_69 = Ghimj[1406];
        W_71 = Ghimj[1407];
        W_73 = Ghimj[1408];
        W_78 = Ghimj[1409];
        W_79 = Ghimj[1410];
        W_81 = Ghimj[1411];
        W_82 = Ghimj[1412];
        W_87 = Ghimj[1413];
        W_88 = Ghimj[1414];
        W_89 = Ghimj[1415];
        W_91 = Ghimj[1416];
        W_92 = Ghimj[1417];
        W_93 = Ghimj[1418];
        W_94 = Ghimj[1419];
        W_96 = Ghimj[1420];
        W_99 = Ghimj[1421];
        W_102 = Ghimj[1422];
        W_103 = Ghimj[1423];
        W_104 = Ghimj[1424];
        W_106 = Ghimj[1425];
        W_107 = Ghimj[1426];
        W_108 = Ghimj[1427];
        W_109 = Ghimj[1428];
        W_110 = Ghimj[1429];
        W_111 = Ghimj[1430];
        W_113 = Ghimj[1431];
        W_114 = Ghimj[1432];
        W_115 = Ghimj[1433];
        W_117 = Ghimj[1434];
        W_119 = Ghimj[1435];
        W_121 = Ghimj[1436];
        W_122 = Ghimj[1437];
        W_124 = Ghimj[1438];
        W_125 = Ghimj[1439];
        W_126 = Ghimj[1440];
        W_127 = Ghimj[1441];
        W_128 = Ghimj[1442];
        W_129 = Ghimj[1443];
        W_130 = Ghimj[1444];
        W_131 = Ghimj[1445];
        W_132 = Ghimj[1446];
        W_133 = Ghimj[1447];
        W_134 = Ghimj[1448];
        W_135 = Ghimj[1449];
        W_136 = Ghimj[1450];
        W_137 = Ghimj[1451];
        W_138 = Ghimj[1452];
        a = - W_46/ Ghimj[272];
        W_46 = -a;
        W_81 = W_81+ a *Ghimj[273];
        W_124 = W_124+ a *Ghimj[274];
        W_137 = W_137+ a *Ghimj[275];
        a = - W_56/ Ghimj[296];
        W_56 = -a;
        W_65 = W_65+ a *Ghimj[297];
        W_81 = W_81+ a *Ghimj[298];
        W_126 = W_126+ a *Ghimj[299];
        a = - W_62/ Ghimj[319];
        W_62 = -a;
        W_93 = W_93+ a *Ghimj[320];
        W_126 = W_126+ a *Ghimj[321];
        W_133 = W_133+ a *Ghimj[322];
        a = - W_65/ Ghimj[331];
        W_65 = -a;
        W_114 = W_114+ a *Ghimj[332];
        W_126 = W_126+ a *Ghimj[333];
        W_132 = W_132+ a *Ghimj[334];
        a = - W_66/ Ghimj[335];
        W_66 = -a;
        W_109 = W_109+ a *Ghimj[336];
        W_126 = W_126+ a *Ghimj[337];
        W_137 = W_137+ a *Ghimj[338];
        a = - W_69/ Ghimj[347];
        W_69 = -a;
        W_93 = W_93+ a *Ghimj[348];
        W_126 = W_126+ a *Ghimj[349];
        W_137 = W_137+ a *Ghimj[350];
        a = - W_71/ Ghimj[356];
        W_71 = -a;
        W_117 = W_117+ a *Ghimj[357];
        W_126 = W_126+ a *Ghimj[358];
        W_137 = W_137+ a *Ghimj[359];
        a = - W_73/ Ghimj[364];
        W_73 = -a;
        W_126 = W_126+ a *Ghimj[365];
        W_135 = W_135+ a *Ghimj[366];
        W_137 = W_137+ a *Ghimj[367];
        a = - W_78/ Ghimj[386];
        W_78 = -a;
        W_103 = W_103+ a *Ghimj[387];
        W_106 = W_106+ a *Ghimj[388];
        W_107 = W_107+ a *Ghimj[389];
        W_110 = W_110+ a *Ghimj[390];
        W_124 = W_124+ a *Ghimj[391];
        W_126 = W_126+ a *Ghimj[392];
        a = - W_79/ Ghimj[393];
        W_79 = -a;
        W_102 = W_102+ a *Ghimj[394];
        W_126 = W_126+ a *Ghimj[395];
        W_137 = W_137+ a *Ghimj[396];
        a = - W_81/ Ghimj[405];
        W_81 = -a;
        W_114 = W_114+ a *Ghimj[406];
        W_124 = W_124+ a *Ghimj[407];
        W_126 = W_126+ a *Ghimj[408];
        W_127 = W_127+ a *Ghimj[409];
        W_129 = W_129+ a *Ghimj[410];
        W_136 = W_136+ a *Ghimj[411];
        a = - W_82/ Ghimj[412];
        W_82 = -a;
        W_113 = W_113+ a *Ghimj[413];
        W_126 = W_126+ a *Ghimj[414];
        W_137 = W_137+ a *Ghimj[415];
        a = - W_87/ Ghimj[444];
        W_87 = -a;
        W_92 = W_92+ a *Ghimj[445];
        W_124 = W_124+ a *Ghimj[446];
        W_126 = W_126+ a *Ghimj[447];
        W_135 = W_135+ a *Ghimj[448];
        W_137 = W_137+ a *Ghimj[449];
        a = - W_88/ Ghimj[450];
        W_88 = -a;
        W_103 = W_103+ a *Ghimj[451];
        W_106 = W_106+ a *Ghimj[452];
        W_124 = W_124+ a *Ghimj[453];
        W_126 = W_126+ a *Ghimj[454];
        W_127 = W_127+ a *Ghimj[455];
        W_137 = W_137+ a *Ghimj[456];
        a = - W_89/ Ghimj[457];
        W_89 = -a;
        W_93 = W_93+ a *Ghimj[458];
        W_94 = W_94+ a *Ghimj[459];
        W_102 = W_102+ a *Ghimj[460];
        W_107 = W_107+ a *Ghimj[461];
        W_109 = W_109+ a *Ghimj[462];
        W_113 = W_113+ a *Ghimj[463];
        W_117 = W_117+ a *Ghimj[464];
        W_124 = W_124+ a *Ghimj[465];
        W_125 = W_125+ a *Ghimj[466];
        W_126 = W_126+ a *Ghimj[467];
        a = - W_91/ Ghimj[481];
        W_91 = -a;
        W_106 = W_106+ a *Ghimj[482];
        W_109 = W_109+ a *Ghimj[483];
        W_126 = W_126+ a *Ghimj[484];
        W_133 = W_133+ a *Ghimj[485];
        W_136 = W_136+ a *Ghimj[486];
        a = - W_92/ Ghimj[489];
        W_92 = -a;
        W_124 = W_124+ a *Ghimj[490];
        W_126 = W_126+ a *Ghimj[491];
        W_133 = W_133+ a *Ghimj[492];
        W_135 = W_135+ a *Ghimj[493];
        W_137 = W_137+ a *Ghimj[494];
        a = - W_93/ Ghimj[497];
        W_93 = -a;
        W_125 = W_125+ a *Ghimj[498];
        W_126 = W_126+ a *Ghimj[499];
        W_133 = W_133+ a *Ghimj[500];
        W_137 = W_137+ a *Ghimj[501];
        a = - W_94/ Ghimj[505];
        W_94 = -a;
        W_125 = W_125+ a *Ghimj[506];
        W_126 = W_126+ a *Ghimj[507];
        W_133 = W_133+ a *Ghimj[508];
        W_137 = W_137+ a *Ghimj[509];
        a = - W_96/ Ghimj[538];
        W_96 = -a;
        W_107 = W_107+ a *Ghimj[539];
        W_108 = W_108+ a *Ghimj[540];
        W_109 = W_109+ a *Ghimj[541];
        W_110 = W_110+ a *Ghimj[542];
        W_113 = W_113+ a *Ghimj[543];
        W_124 = W_124+ a *Ghimj[544];
        W_125 = W_125+ a *Ghimj[545];
        W_126 = W_126+ a *Ghimj[546];
        W_133 = W_133+ a *Ghimj[547];
        W_137 = W_137+ a *Ghimj[548];
        a = - W_99/ Ghimj[565];
        W_99 = -a;
        W_102 = W_102+ a *Ghimj[566];
        W_111 = W_111+ a *Ghimj[567];
        W_125 = W_125+ a *Ghimj[568];
        W_126 = W_126+ a *Ghimj[569];
        W_133 = W_133+ a *Ghimj[570];
        W_137 = W_137+ a *Ghimj[571];
        a = - W_102/ Ghimj[600];
        W_102 = -a;
        W_125 = W_125+ a *Ghimj[601];
        W_126 = W_126+ a *Ghimj[602];
        W_133 = W_133+ a *Ghimj[603];
        W_137 = W_137+ a *Ghimj[604];
        a = - W_103/ Ghimj[605];
        W_103 = -a;
        W_124 = W_124+ a *Ghimj[606];
        W_126 = W_126+ a *Ghimj[607];
        W_127 = W_127+ a *Ghimj[608];
        W_129 = W_129+ a *Ghimj[609];
        a = - W_104/ Ghimj[610];
        W_104 = -a;
        W_125 = W_125+ a *Ghimj[611];
        W_126 = W_126+ a *Ghimj[612];
        W_127 = W_127+ a *Ghimj[613];
        W_129 = W_129+ a *Ghimj[614];
        W_137 = W_137+ a *Ghimj[615];
        a = - W_106/ Ghimj[622];
        W_106 = -a;
        W_124 = W_124+ a *Ghimj[623];
        W_126 = W_126+ a *Ghimj[624];
        W_136 = W_136+ a *Ghimj[625];
        a = - W_107/ Ghimj[626];
        W_107 = -a;
        W_124 = W_124+ a *Ghimj[627];
        W_126 = W_126+ a *Ghimj[628];
        W_136 = W_136+ a *Ghimj[629];
        a = - W_108/ Ghimj[636];
        W_108 = -a;
        W_109 = W_109+ a *Ghimj[637];
        W_113 = W_113+ a *Ghimj[638];
        W_115 = W_115+ a *Ghimj[639];
        W_124 = W_124+ a *Ghimj[640];
        W_125 = W_125+ a *Ghimj[641];
        W_126 = W_126+ a *Ghimj[642];
        W_133 = W_133+ a *Ghimj[643];
        W_135 = W_135+ a *Ghimj[644];
        W_136 = W_136+ a *Ghimj[645];
        W_137 = W_137+ a *Ghimj[646];
        a = - W_109/ Ghimj[648];
        W_109 = -a;
        W_124 = W_124+ a *Ghimj[649];
        W_125 = W_125+ a *Ghimj[650];
        W_126 = W_126+ a *Ghimj[651];
        W_133 = W_133+ a *Ghimj[652];
        W_136 = W_136+ a *Ghimj[653];
        W_137 = W_137+ a *Ghimj[654];
        a = - W_110/ Ghimj[659];
        W_110 = -a;
        W_124 = W_124+ a *Ghimj[660];
        W_125 = W_125+ a *Ghimj[661];
        W_126 = W_126+ a *Ghimj[662];
        W_133 = W_133+ a *Ghimj[663];
        W_136 = W_136+ a *Ghimj[664];
        W_137 = W_137+ a *Ghimj[665];
        a = - W_111/ Ghimj[669];
        W_111 = -a;
        W_115 = W_115+ a *Ghimj[670];
        W_124 = W_124+ a *Ghimj[671];
        W_125 = W_125+ a *Ghimj[672];
        W_126 = W_126+ a *Ghimj[673];
        W_133 = W_133+ a *Ghimj[674];
        W_136 = W_136+ a *Ghimj[675];
        W_137 = W_137+ a *Ghimj[676];
        a = - W_113/ Ghimj[689];
        W_113 = -a;
        W_124 = W_124+ a *Ghimj[690];
        W_125 = W_125+ a *Ghimj[691];
        W_126 = W_126+ a *Ghimj[692];
        W_133 = W_133+ a *Ghimj[693];
        W_135 = W_135+ a *Ghimj[694];
        W_136 = W_136+ a *Ghimj[695];
        W_137 = W_137+ a *Ghimj[696];
        a = - W_114/ Ghimj[697];
        W_114 = -a;
        W_126 = W_126+ a *Ghimj[698];
        W_127 = W_127+ a *Ghimj[699];
        W_129 = W_129+ a *Ghimj[700];
        W_132 = W_132+ a *Ghimj[701];
        W_136 = W_136+ a *Ghimj[702];
        a = - W_115/ Ghimj[706];
        W_115 = -a;
        W_124 = W_124+ a *Ghimj[707];
        W_126 = W_126+ a *Ghimj[708];
        W_127 = W_127+ a *Ghimj[709];
        W_129 = W_129+ a *Ghimj[710];
        W_133 = W_133+ a *Ghimj[711];
        W_136 = W_136+ a *Ghimj[712];
        W_137 = W_137+ a *Ghimj[713];
        a = - W_117/ Ghimj[731];
        W_117 = -a;
        W_121 = W_121+ a *Ghimj[732];
        W_124 = W_124+ a *Ghimj[733];
        W_125 = W_125+ a *Ghimj[734];
        W_126 = W_126+ a *Ghimj[735];
        W_127 = W_127+ a *Ghimj[736];
        W_129 = W_129+ a *Ghimj[737];
        W_133 = W_133+ a *Ghimj[738];
        W_136 = W_136+ a *Ghimj[739];
        W_137 = W_137+ a *Ghimj[740];
        a = - W_119/ Ghimj[767];
        W_119 = -a;
        W_121 = W_121+ a *Ghimj[768];
        W_124 = W_124+ a *Ghimj[769];
        W_125 = W_125+ a *Ghimj[770];
        W_126 = W_126+ a *Ghimj[771];
        W_127 = W_127+ a *Ghimj[772];
        W_129 = W_129+ a *Ghimj[773];
        W_133 = W_133+ a *Ghimj[774];
        W_136 = W_136+ a *Ghimj[775];
        W_137 = W_137+ a *Ghimj[776];
        a = - W_121/ Ghimj[821];
        W_121 = -a;
        W_124 = W_124+ a *Ghimj[822];
        W_125 = W_125+ a *Ghimj[823];
        W_126 = W_126+ a *Ghimj[824];
        W_127 = W_127+ a *Ghimj[825];
        W_129 = W_129+ a *Ghimj[826];
        W_133 = W_133+ a *Ghimj[827];
        W_135 = W_135+ a *Ghimj[828];
        W_136 = W_136+ a *Ghimj[829];
        W_137 = W_137+ a *Ghimj[830];
        a = - W_122/ Ghimj[847];
        W_122 = -a;
        W_124 = W_124+ a *Ghimj[848];
        W_125 = W_125+ a *Ghimj[849];
        W_126 = W_126+ a *Ghimj[850];
        W_127 = W_127+ a *Ghimj[851];
        W_128 = W_128+ a *Ghimj[852];
        W_129 = W_129+ a *Ghimj[853];
        W_130 = W_130+ a *Ghimj[854];
        W_131 = W_131+ a *Ghimj[855];
        W_133 = W_133+ a *Ghimj[856];
        W_135 = W_135+ a *Ghimj[857];
        W_136 = W_136+ a *Ghimj[858];
        W_137 = W_137+ a *Ghimj[859];
        W_138 = W_138+ a *Ghimj[860];
        a = - W_124/ Ghimj[896];
        W_124 = -a;
        W_125 = W_125+ a *Ghimj[897];
        W_126 = W_126+ a *Ghimj[898];
        W_127 = W_127+ a *Ghimj[899];
        W_128 = W_128+ a *Ghimj[900];
        W_129 = W_129+ a *Ghimj[901];
        W_130 = W_130+ a *Ghimj[902];
        W_131 = W_131+ a *Ghimj[903];
        W_132 = W_132+ a *Ghimj[904];
        W_133 = W_133+ a *Ghimj[905];
        W_135 = W_135+ a *Ghimj[906];
        W_136 = W_136+ a *Ghimj[907];
        W_137 = W_137+ a *Ghimj[908];
        W_138 = W_138+ a *Ghimj[909];
        a = - W_125/ Ghimj[934];
        W_125 = -a;
        W_126 = W_126+ a *Ghimj[935];
        W_127 = W_127+ a *Ghimj[936];
        W_128 = W_128+ a *Ghimj[937];
        W_129 = W_129+ a *Ghimj[938];
        W_130 = W_130+ a *Ghimj[939];
        W_131 = W_131+ a *Ghimj[940];
        W_132 = W_132+ a *Ghimj[941];
        W_133 = W_133+ a *Ghimj[942];
        W_134 = W_134+ a *Ghimj[943];
        W_135 = W_135+ a *Ghimj[944];
        W_136 = W_136+ a *Ghimj[945];
        W_137 = W_137+ a *Ghimj[946];
        W_138 = W_138+ a *Ghimj[947];
        a = - W_126/ Ghimj[1023];
        W_126 = -a;
        W_127 = W_127+ a *Ghimj[1024];
        W_128 = W_128+ a *Ghimj[1025];
        W_129 = W_129+ a *Ghimj[1026];
        W_130 = W_130+ a *Ghimj[1027];
        W_131 = W_131+ a *Ghimj[1028];
        W_132 = W_132+ a *Ghimj[1029];
        W_133 = W_133+ a *Ghimj[1030];
        W_134 = W_134+ a *Ghimj[1031];
        W_135 = W_135+ a *Ghimj[1032];
        W_136 = W_136+ a *Ghimj[1033];
        W_137 = W_137+ a *Ghimj[1034];
        W_138 = W_138+ a *Ghimj[1035];
        a = - W_127/ Ghimj[1071];
        W_127 = -a;
        W_128 = W_128+ a *Ghimj[1072];
        W_129 = W_129+ a *Ghimj[1073];
        W_130 = W_130+ a *Ghimj[1074];
        W_131 = W_131+ a *Ghimj[1075];
        W_132 = W_132+ a *Ghimj[1076];
        W_133 = W_133+ a *Ghimj[1077];
        W_134 = W_134+ a *Ghimj[1078];
        W_135 = W_135+ a *Ghimj[1079];
        W_136 = W_136+ a *Ghimj[1080];
        W_137 = W_137+ a *Ghimj[1081];
        W_138 = W_138+ a *Ghimj[1082];
        a = - W_128/ Ghimj[1138];
        W_128 = -a;
        W_129 = W_129+ a *Ghimj[1139];
        W_130 = W_130+ a *Ghimj[1140];
        W_131 = W_131+ a *Ghimj[1141];
        W_132 = W_132+ a *Ghimj[1142];
        W_133 = W_133+ a *Ghimj[1143];
        W_134 = W_134+ a *Ghimj[1144];
        W_135 = W_135+ a *Ghimj[1145];
        W_136 = W_136+ a *Ghimj[1146];
        W_137 = W_137+ a *Ghimj[1147];
        W_138 = W_138+ a *Ghimj[1148];
        a = - W_129/ Ghimj[1176];
        W_129 = -a;
        W_130 = W_130+ a *Ghimj[1177];
        W_131 = W_131+ a *Ghimj[1178];
        W_132 = W_132+ a *Ghimj[1179];
        W_133 = W_133+ a *Ghimj[1180];
        W_134 = W_134+ a *Ghimj[1181];
        W_135 = W_135+ a *Ghimj[1182];
        W_136 = W_136+ a *Ghimj[1183];
        W_137 = W_137+ a *Ghimj[1184];
        W_138 = W_138+ a *Ghimj[1185];
        a = - W_130/ Ghimj[1218];
        W_130 = -a;
        W_131 = W_131+ a *Ghimj[1219];
        W_132 = W_132+ a *Ghimj[1220];
        W_133 = W_133+ a *Ghimj[1221];
        W_134 = W_134+ a *Ghimj[1222];
        W_135 = W_135+ a *Ghimj[1223];
        W_136 = W_136+ a *Ghimj[1224];
        W_137 = W_137+ a *Ghimj[1225];
        W_138 = W_138+ a *Ghimj[1226];
        a = - W_131/ Ghimj[1242];
        W_131 = -a;
        W_132 = W_132+ a *Ghimj[1243];
        W_133 = W_133+ a *Ghimj[1244];
        W_134 = W_134+ a *Ghimj[1245];
        W_135 = W_135+ a *Ghimj[1246];
        W_136 = W_136+ a *Ghimj[1247];
        W_137 = W_137+ a *Ghimj[1248];
        W_138 = W_138+ a *Ghimj[1249];
        a = - W_132/ Ghimj[1262];
        W_132 = -a;
        W_133 = W_133+ a *Ghimj[1263];
        W_134 = W_134+ a *Ghimj[1264];
        W_135 = W_135+ a *Ghimj[1265];
        W_136 = W_136+ a *Ghimj[1266];
        W_137 = W_137+ a *Ghimj[1267];
        W_138 = W_138+ a *Ghimj[1268];
        a = - W_133/ Ghimj[1297];
        W_133 = -a;
        W_134 = W_134+ a *Ghimj[1298];
        W_135 = W_135+ a *Ghimj[1299];
        W_136 = W_136+ a *Ghimj[1300];
        W_137 = W_137+ a *Ghimj[1301];
        W_138 = W_138+ a *Ghimj[1302];
        a = - W_134/ Ghimj[1324];
        W_134 = -a;
        W_135 = W_135+ a *Ghimj[1325];
        W_136 = W_136+ a *Ghimj[1326];
        W_137 = W_137+ a *Ghimj[1327];
        W_138 = W_138+ a *Ghimj[1328];
        a = - W_135/ Ghimj[1370];
        W_135 = -a;
        W_136 = W_136+ a *Ghimj[1371];
        W_137 = W_137+ a *Ghimj[1372];
        W_138 = W_138+ a *Ghimj[1373];
        a = - W_136/ Ghimj[1398];
        W_136 = -a;
        W_137 = W_137+ a *Ghimj[1399];
        W_138 = W_138+ a *Ghimj[1400];
        Ghimj[1401] = W_46;
        Ghimj[1402] = W_56;
        Ghimj[1403] = W_62;
        Ghimj[1404] = W_65;
        Ghimj[1405] = W_66;
        Ghimj[1406] = W_69;
        Ghimj[1407] = W_71;
        Ghimj[1408] = W_73;
        Ghimj[1409] = W_78;
        Ghimj[1410] = W_79;
        Ghimj[1411] = W_81;
        Ghimj[1412] = W_82;
        Ghimj[1413] = W_87;
        Ghimj[1414] = W_88;
        Ghimj[1415] = W_89;
        Ghimj[1416] = W_91;
        Ghimj[1417] = W_92;
        Ghimj[1418] = W_93;
        Ghimj[1419] = W_94;
        Ghimj[1420] = W_96;
        Ghimj[1421] = W_99;
        Ghimj[1422] = W_102;
        Ghimj[1423] = W_103;
        Ghimj[1424] = W_104;
        Ghimj[1425] = W_106;
        Ghimj[1426] = W_107;
        Ghimj[1427] = W_108;
        Ghimj[1428] = W_109;
        Ghimj[1429] = W_110;
        Ghimj[1430] = W_111;
        Ghimj[1431] = W_113;
        Ghimj[1432] = W_114;
        Ghimj[1433] = W_115;
        Ghimj[1434] = W_117;
        Ghimj[1435] = W_119;
        Ghimj[1436] = W_121;
        Ghimj[1437] = W_122;
        Ghimj[1438] = W_124;
        Ghimj[1439] = W_125;
        Ghimj[1440] = W_126;
        Ghimj[1441] = W_127;
        Ghimj[1442] = W_128;
        Ghimj[1443] = W_129;
        Ghimj[1444] = W_130;
        Ghimj[1445] = W_131;
        Ghimj[1446] = W_132;
        Ghimj[1447] = W_133;
        Ghimj[1448] = W_134;
        Ghimj[1449] = W_135;
        Ghimj[1450] = W_136;
        Ghimj[1451] = W_137;
        Ghimj[1452] = W_138;
        W_83 = Ghimj[1453];
        W_88 = Ghimj[1454];
        W_97 = Ghimj[1455];
        W_98 = Ghimj[1456];
        W_103 = Ghimj[1457];
        W_104 = Ghimj[1458];
        W_105 = Ghimj[1459];
        W_106 = Ghimj[1460];
        W_107 = Ghimj[1461];
        W_112 = Ghimj[1462];
        W_114 = Ghimj[1463];
        W_116 = Ghimj[1464];
        W_118 = Ghimj[1465];
        W_119 = Ghimj[1466];
        W_120 = Ghimj[1467];
        W_121 = Ghimj[1468];
        W_122 = Ghimj[1469];
        W_123 = Ghimj[1470];
        W_124 = Ghimj[1471];
        W_125 = Ghimj[1472];
        W_126 = Ghimj[1473];
        W_127 = Ghimj[1474];
        W_128 = Ghimj[1475];
        W_129 = Ghimj[1476];
        W_130 = Ghimj[1477];
        W_131 = Ghimj[1478];
        W_132 = Ghimj[1479];
        W_133 = Ghimj[1480];
        W_134 = Ghimj[1481];
        W_135 = Ghimj[1482];
        W_136 = Ghimj[1483];
        W_137 = Ghimj[1484];
        W_138 = Ghimj[1485];
        a = - W_83/ Ghimj[416];
        W_83 = -a;
        W_128 = W_128+ a *Ghimj[417];
        W_135 = W_135+ a *Ghimj[418];
        W_136 = W_136+ a *Ghimj[419];
        W_138 = W_138+ a *Ghimj[420];
        a = - W_88/ Ghimj[450];
        W_88 = -a;
        W_103 = W_103+ a *Ghimj[451];
        W_106 = W_106+ a *Ghimj[452];
        W_124 = W_124+ a *Ghimj[453];
        W_126 = W_126+ a *Ghimj[454];
        W_127 = W_127+ a *Ghimj[455];
        W_137 = W_137+ a *Ghimj[456];
        a = - W_97/ Ghimj[549];
        W_97 = -a;
        W_98 = W_98+ a *Ghimj[550];
        W_120 = W_120+ a *Ghimj[551];
        W_122 = W_122+ a *Ghimj[552];
        W_126 = W_126+ a *Ghimj[553];
        W_127 = W_127+ a *Ghimj[554];
        W_130 = W_130+ a *Ghimj[555];
        W_137 = W_137+ a *Ghimj[556];
        a = - W_98/ Ghimj[557];
        W_98 = -a;
        W_107 = W_107+ a *Ghimj[558];
        W_120 = W_120+ a *Ghimj[559];
        W_124 = W_124+ a *Ghimj[560];
        W_126 = W_126+ a *Ghimj[561];
        W_127 = W_127+ a *Ghimj[562];
        a = - W_103/ Ghimj[605];
        W_103 = -a;
        W_124 = W_124+ a *Ghimj[606];
        W_126 = W_126+ a *Ghimj[607];
        W_127 = W_127+ a *Ghimj[608];
        W_129 = W_129+ a *Ghimj[609];
        a = - W_104/ Ghimj[610];
        W_104 = -a;
        W_125 = W_125+ a *Ghimj[611];
        W_126 = W_126+ a *Ghimj[612];
        W_127 = W_127+ a *Ghimj[613];
        W_129 = W_129+ a *Ghimj[614];
        W_137 = W_137+ a *Ghimj[615];
        a = - W_105/ Ghimj[616];
        W_105 = -a;
        W_128 = W_128+ a *Ghimj[617];
        W_129 = W_129+ a *Ghimj[618];
        W_132 = W_132+ a *Ghimj[619];
        W_135 = W_135+ a *Ghimj[620];
        W_138 = W_138+ a *Ghimj[621];
        a = - W_106/ Ghimj[622];
        W_106 = -a;
        W_124 = W_124+ a *Ghimj[623];
        W_126 = W_126+ a *Ghimj[624];
        W_136 = W_136+ a *Ghimj[625];
        a = - W_107/ Ghimj[626];
        W_107 = -a;
        W_124 = W_124+ a *Ghimj[627];
        W_126 = W_126+ a *Ghimj[628];
        W_136 = W_136+ a *Ghimj[629];
        a = - W_112/ Ghimj[677];
        W_112 = -a;
        W_116 = W_116+ a *Ghimj[678];
        W_123 = W_123+ a *Ghimj[679];
        W_126 = W_126+ a *Ghimj[680];
        W_128 = W_128+ a *Ghimj[681];
        W_134 = W_134+ a *Ghimj[682];
        W_137 = W_137+ a *Ghimj[683];
        W_138 = W_138+ a *Ghimj[684];
        a = - W_114/ Ghimj[697];
        W_114 = -a;
        W_126 = W_126+ a *Ghimj[698];
        W_127 = W_127+ a *Ghimj[699];
        W_129 = W_129+ a *Ghimj[700];
        W_132 = W_132+ a *Ghimj[701];
        W_136 = W_136+ a *Ghimj[702];
        a = - W_116/ Ghimj[714];
        W_116 = -a;
        W_123 = W_123+ a *Ghimj[715];
        W_127 = W_127+ a *Ghimj[716];
        W_128 = W_128+ a *Ghimj[717];
        W_131 = W_131+ a *Ghimj[718];
        W_134 = W_134+ a *Ghimj[719];
        W_135 = W_135+ a *Ghimj[720];
        W_138 = W_138+ a *Ghimj[721];
        a = - W_118/ Ghimj[745];
        W_118 = -a;
        W_123 = W_123+ a *Ghimj[746];
        W_125 = W_125+ a *Ghimj[747];
        W_126 = W_126+ a *Ghimj[748];
        W_127 = W_127+ a *Ghimj[749];
        W_128 = W_128+ a *Ghimj[750];
        W_129 = W_129+ a *Ghimj[751];
        W_131 = W_131+ a *Ghimj[752];
        W_132 = W_132+ a *Ghimj[753];
        W_134 = W_134+ a *Ghimj[754];
        W_135 = W_135+ a *Ghimj[755];
        W_137 = W_137+ a *Ghimj[756];
        W_138 = W_138+ a *Ghimj[757];
        a = - W_119/ Ghimj[767];
        W_119 = -a;
        W_121 = W_121+ a *Ghimj[768];
        W_124 = W_124+ a *Ghimj[769];
        W_125 = W_125+ a *Ghimj[770];
        W_126 = W_126+ a *Ghimj[771];
        W_127 = W_127+ a *Ghimj[772];
        W_129 = W_129+ a *Ghimj[773];
        W_133 = W_133+ a *Ghimj[774];
        W_136 = W_136+ a *Ghimj[775];
        W_137 = W_137+ a *Ghimj[776];
        a = - W_120/ Ghimj[787];
        W_120 = -a;
        W_122 = W_122+ a *Ghimj[788];
        W_124 = W_124+ a *Ghimj[789];
        W_126 = W_126+ a *Ghimj[790];
        W_127 = W_127+ a *Ghimj[791];
        W_128 = W_128+ a *Ghimj[792];
        W_130 = W_130+ a *Ghimj[793];
        W_133 = W_133+ a *Ghimj[794];
        W_135 = W_135+ a *Ghimj[795];
        W_136 = W_136+ a *Ghimj[796];
        W_137 = W_137+ a *Ghimj[797];
        a = - W_121/ Ghimj[821];
        W_121 = -a;
        W_124 = W_124+ a *Ghimj[822];
        W_125 = W_125+ a *Ghimj[823];
        W_126 = W_126+ a *Ghimj[824];
        W_127 = W_127+ a *Ghimj[825];
        W_129 = W_129+ a *Ghimj[826];
        W_133 = W_133+ a *Ghimj[827];
        W_135 = W_135+ a *Ghimj[828];
        W_136 = W_136+ a *Ghimj[829];
        W_137 = W_137+ a *Ghimj[830];
        a = - W_122/ Ghimj[847];
        W_122 = -a;
        W_124 = W_124+ a *Ghimj[848];
        W_125 = W_125+ a *Ghimj[849];
        W_126 = W_126+ a *Ghimj[850];
        W_127 = W_127+ a *Ghimj[851];
        W_128 = W_128+ a *Ghimj[852];
        W_129 = W_129+ a *Ghimj[853];
        W_130 = W_130+ a *Ghimj[854];
        W_131 = W_131+ a *Ghimj[855];
        W_133 = W_133+ a *Ghimj[856];
        W_135 = W_135+ a *Ghimj[857];
        W_136 = W_136+ a *Ghimj[858];
        W_137 = W_137+ a *Ghimj[859];
        W_138 = W_138+ a *Ghimj[860];
        a = - W_123/ Ghimj[869];
        W_123 = -a;
        W_124 = W_124+ a *Ghimj[870];
        W_125 = W_125+ a *Ghimj[871];
        W_126 = W_126+ a *Ghimj[872];
        W_127 = W_127+ a *Ghimj[873];
        W_128 = W_128+ a *Ghimj[874];
        W_129 = W_129+ a *Ghimj[875];
        W_130 = W_130+ a *Ghimj[876];
        W_131 = W_131+ a *Ghimj[877];
        W_132 = W_132+ a *Ghimj[878];
        W_133 = W_133+ a *Ghimj[879];
        W_134 = W_134+ a *Ghimj[880];
        W_135 = W_135+ a *Ghimj[881];
        W_136 = W_136+ a *Ghimj[882];
        W_137 = W_137+ a *Ghimj[883];
        W_138 = W_138+ a *Ghimj[884];
        a = - W_124/ Ghimj[896];
        W_124 = -a;
        W_125 = W_125+ a *Ghimj[897];
        W_126 = W_126+ a *Ghimj[898];
        W_127 = W_127+ a *Ghimj[899];
        W_128 = W_128+ a *Ghimj[900];
        W_129 = W_129+ a *Ghimj[901];
        W_130 = W_130+ a *Ghimj[902];
        W_131 = W_131+ a *Ghimj[903];
        W_132 = W_132+ a *Ghimj[904];
        W_133 = W_133+ a *Ghimj[905];
        W_135 = W_135+ a *Ghimj[906];
        W_136 = W_136+ a *Ghimj[907];
        W_137 = W_137+ a *Ghimj[908];
        W_138 = W_138+ a *Ghimj[909];
        a = - W_125/ Ghimj[934];
        W_125 = -a;
        W_126 = W_126+ a *Ghimj[935];
        W_127 = W_127+ a *Ghimj[936];
        W_128 = W_128+ a *Ghimj[937];
        W_129 = W_129+ a *Ghimj[938];
        W_130 = W_130+ a *Ghimj[939];
        W_131 = W_131+ a *Ghimj[940];
        W_132 = W_132+ a *Ghimj[941];
        W_133 = W_133+ a *Ghimj[942];
        W_134 = W_134+ a *Ghimj[943];
        W_135 = W_135+ a *Ghimj[944];
        W_136 = W_136+ a *Ghimj[945];
        W_137 = W_137+ a *Ghimj[946];
        W_138 = W_138+ a *Ghimj[947];
        a = - W_126/ Ghimj[1023];
        W_126 = -a;
        W_127 = W_127+ a *Ghimj[1024];
        W_128 = W_128+ a *Ghimj[1025];
        W_129 = W_129+ a *Ghimj[1026];
        W_130 = W_130+ a *Ghimj[1027];
        W_131 = W_131+ a *Ghimj[1028];
        W_132 = W_132+ a *Ghimj[1029];
        W_133 = W_133+ a *Ghimj[1030];
        W_134 = W_134+ a *Ghimj[1031];
        W_135 = W_135+ a *Ghimj[1032];
        W_136 = W_136+ a *Ghimj[1033];
        W_137 = W_137+ a *Ghimj[1034];
        W_138 = W_138+ a *Ghimj[1035];
        a = - W_127/ Ghimj[1071];
        W_127 = -a;
        W_128 = W_128+ a *Ghimj[1072];
        W_129 = W_129+ a *Ghimj[1073];
        W_130 = W_130+ a *Ghimj[1074];
        W_131 = W_131+ a *Ghimj[1075];
        W_132 = W_132+ a *Ghimj[1076];
        W_133 = W_133+ a *Ghimj[1077];
        W_134 = W_134+ a *Ghimj[1078];
        W_135 = W_135+ a *Ghimj[1079];
        W_136 = W_136+ a *Ghimj[1080];
        W_137 = W_137+ a *Ghimj[1081];
        W_138 = W_138+ a *Ghimj[1082];
        a = - W_128/ Ghimj[1138];
        W_128 = -a;
        W_129 = W_129+ a *Ghimj[1139];
        W_130 = W_130+ a *Ghimj[1140];
        W_131 = W_131+ a *Ghimj[1141];
        W_132 = W_132+ a *Ghimj[1142];
        W_133 = W_133+ a *Ghimj[1143];
        W_134 = W_134+ a *Ghimj[1144];
        W_135 = W_135+ a *Ghimj[1145];
        W_136 = W_136+ a *Ghimj[1146];
        W_137 = W_137+ a *Ghimj[1147];
        W_138 = W_138+ a *Ghimj[1148];
        a = - W_129/ Ghimj[1176];
        W_129 = -a;
        W_130 = W_130+ a *Ghimj[1177];
        W_131 = W_131+ a *Ghimj[1178];
        W_132 = W_132+ a *Ghimj[1179];
        W_133 = W_133+ a *Ghimj[1180];
        W_134 = W_134+ a *Ghimj[1181];
        W_135 = W_135+ a *Ghimj[1182];
        W_136 = W_136+ a *Ghimj[1183];
        W_137 = W_137+ a *Ghimj[1184];
        W_138 = W_138+ a *Ghimj[1185];
        a = - W_130/ Ghimj[1218];
        W_130 = -a;
        W_131 = W_131+ a *Ghimj[1219];
        W_132 = W_132+ a *Ghimj[1220];
        W_133 = W_133+ a *Ghimj[1221];
        W_134 = W_134+ a *Ghimj[1222];
        W_135 = W_135+ a *Ghimj[1223];
        W_136 = W_136+ a *Ghimj[1224];
        W_137 = W_137+ a *Ghimj[1225];
        W_138 = W_138+ a *Ghimj[1226];
        a = - W_131/ Ghimj[1242];
        W_131 = -a;
        W_132 = W_132+ a *Ghimj[1243];
        W_133 = W_133+ a *Ghimj[1244];
        W_134 = W_134+ a *Ghimj[1245];
        W_135 = W_135+ a *Ghimj[1246];
        W_136 = W_136+ a *Ghimj[1247];
        W_137 = W_137+ a *Ghimj[1248];
        W_138 = W_138+ a *Ghimj[1249];
        a = - W_132/ Ghimj[1262];
        W_132 = -a;
        W_133 = W_133+ a *Ghimj[1263];
        W_134 = W_134+ a *Ghimj[1264];
        W_135 = W_135+ a *Ghimj[1265];
        W_136 = W_136+ a *Ghimj[1266];
        W_137 = W_137+ a *Ghimj[1267];
        W_138 = W_138+ a *Ghimj[1268];
        a = - W_133/ Ghimj[1297];
        W_133 = -a;
        W_134 = W_134+ a *Ghimj[1298];
        W_135 = W_135+ a *Ghimj[1299];
        W_136 = W_136+ a *Ghimj[1300];
        W_137 = W_137+ a *Ghimj[1301];
        W_138 = W_138+ a *Ghimj[1302];
        a = - W_134/ Ghimj[1324];
        W_134 = -a;
        W_135 = W_135+ a *Ghimj[1325];
        W_136 = W_136+ a *Ghimj[1326];
        W_137 = W_137+ a *Ghimj[1327];
        W_138 = W_138+ a *Ghimj[1328];
        a = - W_135/ Ghimj[1370];
        W_135 = -a;
        W_136 = W_136+ a *Ghimj[1371];
        W_137 = W_137+ a *Ghimj[1372];
        W_138 = W_138+ a *Ghimj[1373];
        a = - W_136/ Ghimj[1398];
        W_136 = -a;
        W_137 = W_137+ a *Ghimj[1399];
        W_138 = W_138+ a *Ghimj[1400];
        a = - W_137/ Ghimj[1451];
        W_137 = -a;
        W_138 = W_138+ a *Ghimj[1452];
        Ghimj[1453] = W_83;
        Ghimj[1454] = W_88;
        Ghimj[1455] = W_97;
        Ghimj[1456] = W_98;
        Ghimj[1457] = W_103;
        Ghimj[1458] = W_104;
        Ghimj[1459] = W_105;
        Ghimj[1460] = W_106;
        Ghimj[1461] = W_107;
        Ghimj[1462] = W_112;
        Ghimj[1463] = W_114;
        Ghimj[1464] = W_116;
        Ghimj[1465] = W_118;
        Ghimj[1466] = W_119;
        Ghimj[1467] = W_120;
        Ghimj[1468] = W_121;
        Ghimj[1469] = W_122;
        Ghimj[1470] = W_123;
        Ghimj[1471] = W_124;
        Ghimj[1472] = W_125;
        Ghimj[1473] = W_126;
        Ghimj[1474] = W_127;
        Ghimj[1475] = W_128;
        Ghimj[1476] = W_129;
        Ghimj[1477] = W_130;
        Ghimj[1478] = W_131;
        Ghimj[1479] = W_132;
        Ghimj[1480] = W_133;
        Ghimj[1481] = W_134;
        Ghimj[1482] = W_135;
        Ghimj[1483] = W_136;
        Ghimj[1484] = W_137;
        Ghimj[1485] = W_138;
}

__device__ void ros_Decomp(double * __restrict__ Ghimj, int &Ndec, int VL_GLO)
{
    kppDecomp(Ghimj, VL_GLO);
    Ndec++;
}


__device__ void ros_PrepareMatrix(double &H, int direction, double gam, double *jac0, double *Ghimj,  int &Nsng, int &Ndec, int VL_GLO)
{
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    int ising, nConsecutive;
    double ghinv;
    
        ghinv = ONE/(direction*H*gam);
        for (int i=0; i<LU_NONZERO; i++)
            Ghimj[i] = -jac0[i];

        Ghimj[0] += ghinv;
        Ghimj[1] += ghinv;
        Ghimj[2] += ghinv;
        Ghimj[3] += ghinv;
        Ghimj[4] += ghinv;
        Ghimj[5] += ghinv;
        Ghimj[6] += ghinv;
        Ghimj[9] += ghinv;
        Ghimj[25] += ghinv;
        Ghimj[29] += ghinv;
        Ghimj[38] += ghinv;
        Ghimj[43] += ghinv;
        Ghimj[46] += ghinv;
        Ghimj[48] += ghinv;
        Ghimj[52] += ghinv;
        Ghimj[58] += ghinv;
        Ghimj[60] += ghinv;
        Ghimj[62] += ghinv;
        Ghimj[64] += ghinv;
        Ghimj[68] += ghinv;
        Ghimj[69] += ghinv;
        Ghimj[72] += ghinv;
        Ghimj[75] += ghinv;
        Ghimj[112] += ghinv;
        Ghimj[123] += ghinv;
        Ghimj[140] += ghinv;
        Ghimj[148] += ghinv;
        Ghimj[163] += ghinv;
        Ghimj[170] += ghinv;
        Ghimj[182] += ghinv;
        Ghimj[185] += ghinv;
        Ghimj[190] += ghinv;
        Ghimj[194] += ghinv;
        Ghimj[202] += ghinv;
        Ghimj[206] += ghinv;
        Ghimj[233] += ghinv;
        Ghimj[244] += ghinv;
        Ghimj[251] += ghinv;
        Ghimj[255] += ghinv;
        Ghimj[258] += ghinv;
        Ghimj[260] += ghinv;
        Ghimj[262] += ghinv;
        Ghimj[264] += ghinv;
        Ghimj[266] += ghinv;
        Ghimj[268] += ghinv;
        Ghimj[270] += ghinv;
        Ghimj[272] += ghinv;
        Ghimj[276] += ghinv;
        Ghimj[278] += ghinv;
        Ghimj[280] += ghinv;
        Ghimj[282] += ghinv;
        Ghimj[285] += ghinv;
        Ghimj[288] += ghinv;
        Ghimj[290] += ghinv;
        Ghimj[292] += ghinv;
        Ghimj[294] += ghinv;
        Ghimj[296] += ghinv;
        Ghimj[300] += ghinv;
        Ghimj[303] += ghinv;
        Ghimj[306] += ghinv;
        Ghimj[310] += ghinv;
        Ghimj[315] += ghinv;
        Ghimj[319] += ghinv;
        Ghimj[323] += ghinv;
        Ghimj[327] += ghinv;
        Ghimj[331] += ghinv;
        Ghimj[335] += ghinv;
        Ghimj[339] += ghinv;
        Ghimj[343] += ghinv;
        Ghimj[347] += ghinv;
        Ghimj[352] += ghinv;
        Ghimj[356] += ghinv;
        Ghimj[360] += ghinv;
        Ghimj[364] += ghinv;
        Ghimj[368] += ghinv;
        Ghimj[374] += ghinv;
        Ghimj[377] += ghinv;
        Ghimj[382] += ghinv;
        Ghimj[386] += ghinv;
        Ghimj[393] += ghinv;
        Ghimj[397] += ghinv;
        Ghimj[405] += ghinv;
        Ghimj[412] += ghinv;
        Ghimj[416] += ghinv;
        Ghimj[421] += ghinv;
        Ghimj[427] += ghinv;
        Ghimj[436] += ghinv;
        Ghimj[444] += ghinv;
        Ghimj[450] += ghinv;
        Ghimj[457] += ghinv;
        Ghimj[469] += ghinv;
        Ghimj[481] += ghinv;
        Ghimj[489] += ghinv;
        Ghimj[497] += ghinv;
        Ghimj[505] += ghinv;
        Ghimj[514] += ghinv;
        Ghimj[538] += ghinv;
        Ghimj[549] += ghinv;
        Ghimj[557] += ghinv;
        Ghimj[565] += ghinv;
        Ghimj[573] += ghinv;
        Ghimj[586] += ghinv;
        Ghimj[600] += ghinv;
        Ghimj[605] += ghinv;
        Ghimj[610] += ghinv;
        Ghimj[616] += ghinv;
        Ghimj[622] += ghinv;
        Ghimj[626] += ghinv;
        Ghimj[636] += ghinv;
        Ghimj[648] += ghinv;
        Ghimj[659] += ghinv;
        Ghimj[669] += ghinv;
        Ghimj[677] += ghinv;
        Ghimj[689] += ghinv;
        Ghimj[697] += ghinv;
        Ghimj[706] += ghinv;
        Ghimj[714] += ghinv;
        Ghimj[731] += ghinv;
        Ghimj[745] += ghinv;
        Ghimj[767] += ghinv;
        Ghimj[787] += ghinv;
        Ghimj[821] += ghinv;
        Ghimj[847] += ghinv;
        Ghimj[869] += ghinv;
        Ghimj[896] += ghinv;
        Ghimj[934] += ghinv;
        Ghimj[1023] += ghinv;
        Ghimj[1071] += ghinv;
        Ghimj[1138] += ghinv;
        Ghimj[1176] += ghinv;
        Ghimj[1218] += ghinv;
        Ghimj[1242] += ghinv;
        Ghimj[1262] += ghinv;
        Ghimj[1297] += ghinv;
        Ghimj[1324] += ghinv;
        Ghimj[1370] += ghinv;
        Ghimj[1398] += ghinv;
        Ghimj[1451] += ghinv;
        Ghimj[1485] += ghinv;
        Ghimj[1486] += ghinv;
        ros_Decomp(Ghimj, Ndec, VL_GLO);
}

__device__ void Jac_sp(const double * __restrict__ var, const double * __restrict__ fix,
                 const double * __restrict__ rconst, double * __restrict__ jcb, int &Njac, const int VL_GLO)
{
    int index = blockIdx.x*blockDim.x+threadIdx.x;

 double dummy, B_0, B_1, B_2, B_3, B_4, B_5, B_6, B_7, B_8, B_9, B_10, B_11, B_12, B_13, B_14, B_15, B_16, B_17, B_18, B_19, B_20, B_21, B_22, B_23, B_24, B_25, B_26, B_27, B_28, B_29, B_30, B_31, B_32, B_33, B_34, B_35, B_36, B_37, B_38, B_39, B_40, B_41, B_42, B_43, B_44, B_45, B_46, B_47, B_48, B_49, B_50, B_51, B_52, B_53, B_54, B_55, B_56, B_57, B_58, B_59, B_60, B_61, B_62, B_63, B_64, B_65, B_66, B_67, B_68, B_69, B_70, B_71, B_72, B_73, B_74, B_75, B_76, B_77, B_78, B_79, B_80, B_81, B_82, B_83, B_84, B_85, B_86, B_87, B_88, B_89, B_90, B_91, B_92, B_93, B_94, B_95, B_96, B_97, B_98, B_99, B_100, B_101, B_102, B_103, B_104, B_105, B_106, B_107, B_108, B_109, B_110, B_111, B_112, B_113, B_114, B_115, B_116, B_117, B_118, B_119, B_120, B_121, B_122, B_123, B_124, B_125, B_126, B_127, B_128, B_129, B_130, B_131, B_132, B_133, B_134, B_135, B_136, B_137, B_138, B_139, B_140, B_141, B_142, B_143, B_144, B_145, B_146, B_147, B_148, B_149, B_150, B_151, B_152, B_153, B_154, B_155, B_156, B_157, B_158, B_159, B_160, B_161, B_162, B_163, B_164, B_165, B_166, B_167, B_168, B_169, B_170, B_171, B_172, B_173, B_174, B_175, B_176, B_177, B_178, B_179, B_180, B_181, B_182, B_183, B_184, B_185, B_186, B_187, B_188, B_189, B_190, B_191, B_192, B_193, B_194, B_195, B_196, B_197, B_198, B_199, B_200, B_201, B_202, B_203, B_204, B_205, B_206, B_207, B_208, B_209, B_210, B_211, B_212, B_213, B_214, B_215, B_216, B_217, B_218, B_219, B_220, B_221, B_222, B_223, B_224, B_225, B_226, B_227, B_228, B_229, B_230, B_231, B_232, B_233, B_234, B_235, B_236, B_237, B_238, B_239, B_240, B_241, B_242, B_243, B_244, B_245, B_246, B_247, B_248, B_249, B_250, B_251, B_252, B_253, B_254, B_255, B_256, B_257, B_258, B_259, B_260, B_261, B_262, B_263, B_264, B_265, B_266, B_267, B_268, B_269, B_270, B_271, B_272, B_273, B_274, B_275, B_276, B_277, B_278, B_279, B_280, B_281, B_282, B_283, B_284, B_285, B_286, B_287, B_288, B_289, B_290, B_291, B_292, B_293, B_294, B_295, B_296, B_297, B_298, B_299, B_300, B_301, B_302, B_303, B_304, B_305, B_306, B_307, B_308, B_309, B_310, B_311, B_312, B_313, B_314, B_315, B_316, B_317, B_318, B_319, B_320, B_321, B_322, B_323, B_324, B_325, B_326, B_327, B_328, B_329, B_330, B_331, B_332, B_333, B_334, B_335, B_336, B_337, B_338, B_339, B_340, B_341, B_342, B_343, B_344, B_345, B_346, B_347, B_348, B_349, B_350, B_351, B_352, B_353, B_354, B_355, B_356, B_357, B_358, B_359, B_360, B_361, B_362, B_363, B_364, B_365, B_366, B_367, B_368, B_369, B_370, B_371, B_372, B_373, B_374, B_375, B_376, B_377, B_378, B_379, B_380, B_381, B_382, B_383, B_384, B_385, B_386, B_387, B_388, B_389, B_390, B_391, B_392, B_393, B_394, B_395, B_396, B_397, B_398, B_399, B_400, B_401, B_402, B_403, B_404, B_405, B_406, B_407, B_408, B_409, B_410, B_411, B_412, B_413, B_414, B_415, B_416, B_417, B_418, B_419, B_420, B_421, B_422, B_423, B_424, B_425, B_426, B_427, B_428, B_429, B_430, B_431, B_432, B_433, B_434, B_435, B_436, B_437, B_438, B_439, B_440, B_441, B_442, B_443, B_444, B_445, B_446, B_447, B_448, B_449, B_450, B_451, B_452, B_453, B_454, B_455, B_456, B_457, B_458, B_459, B_460, B_461, B_462, B_463, B_464, B_465, B_466, B_467, B_468, B_469, B_470, B_471, B_472, B_473, B_474, B_475, B_476, B_477, B_478, B_479, B_480, B_481, B_482, B_483, B_484, B_485, B_486, B_487, B_488, B_489, B_490, B_491, B_492, B_493, B_494, B_495, B_496, B_497, B_498, B_499, B_500, B_501, B_502, B_503, B_504, B_505, B_506, B_507, B_508, B_509, B_510, B_511, B_512, B_513, B_514, B_515, B_516, B_517, B_518, B_519, B_520, B_521, B_522;


    Njac++;

        B_0 = rconst(index,0)*fix[0];
        B_2 = rconst(index,1)*fix[0];
        B_4 = 1.2e-10*var[124];
        B_5 = 1.2e-10*var[120];
        B_6 = rconst(index,3)*var[131];
        B_7 = rconst(index,3)*var[124];
        B_8 = rconst(index,4)*fix[0];
        B_10 = rconst(index,5)*var[124];
        B_11 = rconst(index,5)*var[122];
        B_12 = 1.2e-10*var[120];
        B_13 = 1.2e-10*var[97];
        B_14 = rconst(index,7)*var[131];
        B_15 = rconst(index,7)*var[126];
        B_16 = rconst(index,8)*var[126];
        B_17 = rconst(index,8)*var[124];
        B_18 = rconst(index,9)*var[126];
        B_19 = rconst(index,9)*var[97];
        B_20 = rconst(index,10)*var[137];
        B_21 = rconst(index,10)*var[131];
        B_22 = rconst(index,11)*var[137];
        B_23 = rconst(index,11)*var[124];
        B_24 = 7.2e-11*var[137];
        B_25 = 7.2e-11*var[122];
        B_26 = 6.9e-12*var[137];
        B_27 = 6.9e-12*var[122];
        B_28 = 1.6e-12*var[137];
        B_29 = 1.6e-12*var[122];
        B_30 = rconst(index,15)*var[137];
        B_31 = rconst(index,15)*var[126];
        B_32 = rconst(index,16)*2*var[137];
        B_33 = rconst(index,17)*var[128];
        B_34 = rconst(index,17)*var[120];
        B_35 = 1.8e-12*var[126];
        B_36 = 1.8e-12*var[88];
        B_37 = rconst(index,19)*fix[0];
        B_39 = rconst(index,20)*fix[1];
        B_41 = rconst(index,21)*var[120];
        B_42 = rconst(index,21)*var[60];
        B_43 = rconst(index,22)*var[120];
        B_44 = rconst(index,22)*var[60];
        B_45 = rconst(index,23)*var[133];
        B_46 = rconst(index,23)*var[124];
        B_47 = rconst(index,24)*var[133];
        B_48 = rconst(index,24)*var[59];
        B_49 = rconst(index,25)*var[135];
        B_50 = rconst(index,25)*var[131];
        B_51 = rconst(index,26)*var[135];
        B_52 = rconst(index,26)*var[124];
        B_53 = rconst(index,27)*var[135];
        B_54 = rconst(index,27)*var[59];
        B_55 = rconst(index,28)*var[136];
        B_56 = rconst(index,28)*var[133];
        B_57 = rconst(index,29)*var[136];
        B_58 = rconst(index,29)*var[135];
        B_59 = rconst(index,30);
        B_60 = rconst(index,31)*var[133];
        B_61 = rconst(index,31)*var[126];
        B_62 = rconst(index,32)*var[137];
        B_63 = rconst(index,32)*var[133];
        B_64 = rconst(index,33)*var[135];
        B_65 = rconst(index,33)*var[126];
        B_66 = rconst(index,34)*var[137];
        B_67 = rconst(index,34)*var[135];
        B_68 = 3.5e-12*var[137];
        B_69 = 3.5e-12*var[136];
        B_70 = rconst(index,36)*var[126];
        B_71 = rconst(index,36)*var[76];
        B_72 = rconst(index,37)*var[126];
        B_73 = rconst(index,37)*var[101];
        B_74 = rconst(index,38);
        B_75 = rconst(index,39)*var[126];
        B_76 = rconst(index,39)*var[73];
        B_77 = rconst(index,40)*var[126];
        B_78 = rconst(index,40)*var[47];
        B_79 = rconst(index,41)*var[124];
        B_80 = rconst(index,41)*var[92];
        B_81 = rconst(index,42)*var[137];
        B_82 = rconst(index,42)*var[92];
        B_83 = rconst(index,43)*var[137];
        B_84 = rconst(index,43)*var[92];
        B_85 = rconst(index,44)*var[133];
        B_86 = rconst(index,44)*var[92];
        B_87 = rconst(index,45)*var[133];
        B_88 = rconst(index,45)*var[92];
        B_89 = rconst(index,46)*var[135];
        B_90 = rconst(index,46)*var[92];
        B_91 = rconst(index,47)*var[135];
        B_92 = rconst(index,47)*var[92];
        B_93 = 1.2e-14*var[124];
        B_94 = 1.2e-14*var[84];
        B_95 = 1300;
        B_96 = rconst(index,50)*var[126];
        B_97 = rconst(index,50)*var[87];
        B_98 = rconst(index,51)*var[87];
        B_99 = rconst(index,51)*var[70];
        B_100 = rconst(index,52)*var[135];
        B_101 = rconst(index,52)*var[87];
        B_102 = 1.66e-12*var[126];
        B_103 = 1.66e-12*var[70];
        B_104 = rconst(index,54)*var[126];
        B_105 = rconst(index,54)*var[61];
        B_106 = rconst(index,55)*fix[0];
        B_108 = 1.75e-10*var[120];
        B_109 = 1.75e-10*var[98];
        B_110 = rconst(index,57)*var[126];
        B_111 = rconst(index,57)*var[98];
        B_112 = rconst(index,58)*var[126];
        B_113 = rconst(index,58)*var[89];
        B_114 = rconst(index,59)*var[137];
        B_115 = rconst(index,59)*var[125];
        B_116 = rconst(index,60)*var[133];
        B_117 = rconst(index,60)*var[125];
        B_118 = 1.3e-12*var[136];
        B_119 = 1.3e-12*var[125];
        B_120 = rconst(index,62)*2*var[125];
        B_121 = rconst(index,63)*2*var[125];
        B_122 = rconst(index,64)*var[126];
        B_123 = rconst(index,64)*var[104];
        B_124 = rconst(index,65)*var[130];
        B_125 = rconst(index,65)*var[126];
        B_126 = rconst(index,66)*var[136];
        B_127 = rconst(index,66)*var[130];
        B_128 = rconst(index,67)*var[126];
        B_129 = rconst(index,67)*var[95];
        B_130 = 4e-13*var[126];
        B_131 = 4e-13*var[78];
        B_132 = rconst(index,69)*var[126];
        B_133 = rconst(index,69)*var[48];
        B_134 = rconst(index,70)*var[124];
        B_135 = rconst(index,70)*var[103];
        B_136 = rconst(index,71)*var[126];
        B_137 = rconst(index,71)*var[103];
        B_138 = rconst(index,72)*var[137];
        B_139 = rconst(index,72)*var[117];
        B_140 = rconst(index,73)*var[133];
        B_141 = rconst(index,73)*var[117];
        B_142 = 2.3e-12*var[136];
        B_143 = 2.3e-12*var[117];
        B_144 = rconst(index,75)*var[125];
        B_145 = rconst(index,75)*var[117];
        B_146 = rconst(index,76)*var[126];
        B_147 = rconst(index,76)*var[71];
        B_148 = rconst(index,77)*var[126];
        B_149 = rconst(index,77)*var[119];
        B_150 = rconst(index,78)*var[136];
        B_151 = rconst(index,78)*var[119];
        B_152 = rconst(index,79)*var[126];
        B_153 = rconst(index,79)*var[74];
        B_154 = rconst(index,80)*var[137];
        B_155 = rconst(index,80)*var[121];
        B_156 = rconst(index,81)*var[137];
        B_157 = rconst(index,81)*var[121];
        B_158 = rconst(index,82)*var[133];
        B_159 = rconst(index,82)*var[121];
        B_160 = rconst(index,83)*var[135];
        B_161 = rconst(index,83)*var[121];
        B_162 = 4e-12*var[136];
        B_163 = 4e-12*var[121];
        B_164 = rconst(index,85)*var[125];
        B_165 = rconst(index,85)*var[121];
        B_166 = rconst(index,86)*var[125];
        B_167 = rconst(index,86)*var[121];
        B_168 = rconst(index,87)*var[121];
        B_169 = rconst(index,87)*var[117];
        B_170 = rconst(index,88)*2*var[121];
        B_171 = rconst(index,89)*var[126];
        B_172 = rconst(index,89)*var[63];
        B_173 = rconst(index,90)*var[126];
        B_174 = rconst(index,90)*var[58];
        B_175 = rconst(index,91)*var[126];
        B_176 = rconst(index,91)*var[77];
        B_177 = rconst(index,92);
        B_178 = rconst(index,93)*var[126];
        B_179 = rconst(index,93)*var[49];
        B_180 = rconst(index,94)*var[124];
        B_181 = rconst(index,94)*var[107];
        B_182 = rconst(index,95)*var[126];
        B_183 = rconst(index,95)*var[107];
        B_184 = rconst(index,96)*var[136];
        B_185 = rconst(index,96)*var[107];
        B_186 = rconst(index,97)*var[137];
        B_187 = rconst(index,97)*var[93];
        B_188 = rconst(index,98)*var[133];
        B_189 = rconst(index,98)*var[93];
        B_190 = rconst(index,99)*var[125];
        B_191 = rconst(index,99)*var[93];
        B_192 = rconst(index,100)*var[126];
        B_193 = rconst(index,100)*var[69];
        B_194 = rconst(index,101)*var[137];
        B_195 = rconst(index,101)*var[115];
        B_196 = rconst(index,102)*var[133];
        B_197 = rconst(index,102)*var[115];
        B_198 = rconst(index,103)*var[126];
        B_199 = rconst(index,103)*var[67];
        B_200 = rconst(index,104)*var[126];
        B_201 = rconst(index,104)*var[86];
        B_202 = rconst(index,105)*var[137];
        B_203 = rconst(index,105)*var[94];
        B_204 = rconst(index,106)*var[133];
        B_205 = rconst(index,106)*var[94];
        B_206 = rconst(index,107)*var[125];
        B_207 = rconst(index,107)*var[94];
        B_208 = rconst(index,108)*var[126];
        B_209 = rconst(index,108)*var[72];
        B_210 = rconst(index,109)*var[126];
        B_211 = rconst(index,109)*var[108];
        B_212 = rconst(index,110)*var[126];
        B_213 = rconst(index,110)*var[96];
        B_214 = rconst(index,111)*var[126];
        B_215 = rconst(index,111)*var[62];
        B_216 = rconst(index,112)*var[126];
        B_217 = rconst(index,112)*var[40];
        B_218 = rconst(index,113)*var[125];
        B_219 = rconst(index,113)*var[102];
        B_220 = rconst(index,114)*var[137];
        B_221 = rconst(index,114)*var[102];
        B_222 = rconst(index,115)*var[133];
        B_223 = rconst(index,115)*var[102];
        B_224 = rconst(index,116)*var[126];
        B_225 = rconst(index,116)*var[79];
        B_226 = rconst(index,117)*var[124];
        B_227 = rconst(index,117)*var[110];
        B_228 = rconst(index,118)*var[126];
        B_229 = rconst(index,118)*var[110];
        B_230 = rconst(index,119)*var[137];
        B_231 = rconst(index,119)*var[113];
        B_232 = rconst(index,120)*var[133];
        B_233 = rconst(index,120)*var[113];
        B_234 = rconst(index,121)*var[135];
        B_235 = rconst(index,121)*var[113];
        B_236 = 2e-12*var[125];
        B_237 = 2e-12*var[113];
        B_238 = 2e-12*2*var[113];
        B_239 = 3e-11*var[126];
        B_240 = 3e-11*var[82];
        B_241 = rconst(index,125)*var[126];
        B_242 = rconst(index,125)*var[85];
        B_243 = rconst(index,126)*var[137];
        B_244 = rconst(index,126)*var[99];
        B_245 = rconst(index,127)*var[133];
        B_246 = rconst(index,127)*var[99];
        B_247 = rconst(index,128)*var[126];
        B_248 = rconst(index,128)*var[68];
        B_249 = 1.7e-12*var[126];
        B_250 = 1.7e-12*var[111];
        B_251 = 3.2e-11*var[126];
        B_252 = 3.2e-11*var[64];
        B_253 = rconst(index,131);
        B_254 = rconst(index,132)*var[124];
        B_255 = rconst(index,132)*var[106];
        B_256 = rconst(index,133)*var[126];
        B_257 = rconst(index,133)*var[106];
        B_258 = rconst(index,134)*var[136];
        B_259 = rconst(index,134)*var[106];
        B_260 = rconst(index,135)*var[137];
        B_261 = rconst(index,135)*var[109];
        B_262 = rconst(index,136)*var[133];
        B_263 = rconst(index,136)*var[109];
        B_264 = 2e-12*var[125];
        B_265 = 2e-12*var[109];
        B_266 = 2e-12*2*var[109];
        B_267 = 1e-10*var[126];
        B_268 = 1e-10*var[66];
        B_269 = 1.3e-11*var[126];
        B_270 = 1.3e-11*var[91];
        B_271 = rconst(index,141)*var[127];
        B_272 = rconst(index,141)*var[124];
        B_273 = rconst(index,142)*var[134];
        B_274 = rconst(index,142)*var[131];
        B_275 = rconst(index,143)*2*var[134];
        B_276 = rconst(index,144)*2*var[134];
        B_277 = rconst(index,145)*2*var[134];
        B_278 = rconst(index,146)*2*var[134];
        B_279 = rconst(index,147);
        B_280 = rconst(index,148)*var[127];
        B_281 = rconst(index,148)*var[97];
        B_282 = rconst(index,149)*var[137];
        B_283 = rconst(index,149)*var[127];
        B_284 = rconst(index,150)*var[137];
        B_285 = rconst(index,150)*var[127];
        B_286 = rconst(index,151)*var[127];
        B_287 = rconst(index,151)*var[88];
        B_288 = rconst(index,152)*var[134];
        B_289 = rconst(index,152)*var[126];
        B_290 = rconst(index,153)*var[137];
        B_291 = rconst(index,153)*var[134];
        B_292 = rconst(index,154)*var[138];
        B_293 = rconst(index,154)*var[126];
        B_294 = rconst(index,155)*var[126];
        B_295 = rconst(index,155)*var[112];
        B_296 = rconst(index,156)*var[134];
        B_297 = rconst(index,156)*var[133];
        B_298 = rconst(index,157)*var[135];
        B_299 = rconst(index,157)*var[134];
        B_300 = rconst(index,158);
        B_301 = rconst(index,159)*var[131];
        B_302 = rconst(index,159)*var[116];
        B_303 = rconst(index,160)*var[127];
        B_304 = rconst(index,160)*var[116];
        B_305 = rconst(index,161)*var[127];
        B_306 = rconst(index,161)*var[98];
        B_307 = rconst(index,162)*var[130];
        B_308 = rconst(index,162)*var[127];
        B_309 = 5.9e-11*var[127];
        B_310 = 5.9e-11*var[104];
        B_311 = rconst(index,164)*var[134];
        B_312 = rconst(index,164)*var[125];
        B_313 = 3.3e-10*var[120];
        B_314 = 3.3e-10*var[41];
        B_315 = 1.65e-10*var[120];
        B_316 = 1.65e-10*var[75];
        B_317 = rconst(index,167)*var[126];
        B_318 = rconst(index,167)*var[75];
        B_319 = 3.25e-10*var[120];
        B_320 = 3.25e-10*var[57];
        B_321 = rconst(index,169)*var[126];
        B_322 = rconst(index,169)*var[57];
        B_323 = rconst(index,170)*var[127];
        B_324 = rconst(index,170)*var[103];
        B_325 = 8e-11*var[127];
        B_326 = 8e-11*var[119];
        B_327 = 1.4e-10*var[120];
        B_328 = 1.4e-10*var[42];
        B_329 = 2.3e-10*var[120];
        B_330 = 2.3e-10*var[43];
        B_331 = rconst(index,174)*var[129];
        B_332 = rconst(index,174)*var[124];
        B_333 = rconst(index,175)*var[132];
        B_334 = rconst(index,175)*var[131];
        B_335 = 2.7e-12*2*var[132];
        B_336 = rconst(index,177)*2*var[132];
        B_337 = rconst(index,178)*var[137];
        B_338 = rconst(index,178)*var[129];
        B_339 = rconst(index,179)*var[137];
        B_340 = rconst(index,179)*var[132];
        B_341 = rconst(index,180)*var[126];
        B_342 = rconst(index,180)*var[123];
        B_343 = rconst(index,181)*var[131];
        B_344 = rconst(index,181)*var[118];
        B_345 = rconst(index,182)*var[126];
        B_346 = rconst(index,182)*var[100];
        B_347 = 4.9e-11*var[129];
        B_348 = 4.9e-11*var[105];
        B_349 = rconst(index,184)*var[133];
        B_350 = rconst(index,184)*var[132];
        B_351 = rconst(index,185)*var[135];
        B_352 = rconst(index,185)*var[132];
        B_353 = rconst(index,186);
        B_354 = rconst(index,187)*var[130];
        B_355 = rconst(index,187)*var[129];
        B_356 = rconst(index,188)*var[129];
        B_357 = rconst(index,188)*var[104];
        B_358 = rconst(index,189)*var[132];
        B_359 = rconst(index,189)*var[125];
        B_360 = rconst(index,190)*var[132];
        B_361 = rconst(index,190)*var[125];
        B_362 = rconst(index,191)*var[126];
        B_363 = rconst(index,191)*var[53];
        B_364 = rconst(index,192)*var[129];
        B_365 = rconst(index,192)*var[103];
        B_366 = rconst(index,193)*var[129];
        B_367 = rconst(index,193)*var[119];
        B_368 = rconst(index,194)*var[126];
        B_369 = rconst(index,194)*var[45];
        B_370 = rconst(index,195)*var[126];
        B_371 = rconst(index,195)*var[44];
        B_372 = 3.32e-15*var[129];
        B_373 = 3.32e-15*var[90];
        B_374 = 1.1e-15*var[129];
        B_375 = 1.1e-15*var[80];
        B_376 = rconst(index,198)*var[127];
        B_377 = rconst(index,198)*var[100];
        B_378 = rconst(index,199)*var[134];
        B_379 = rconst(index,199)*var[132];
        B_380 = rconst(index,200)*var[134];
        B_381 = rconst(index,200)*var[132];
        B_382 = rconst(index,201)*var[134];
        B_383 = rconst(index,201)*var[132];
        B_384 = 1.45e-11*var[127];
        B_385 = 1.45e-11*var[90];
        B_386 = rconst(index,203)*var[126];
        B_387 = rconst(index,203)*var[54];
        B_388 = rconst(index,204)*var[126];
        B_389 = rconst(index,204)*var[55];
        B_390 = rconst(index,205)*var[126];
        B_391 = rconst(index,205)*var[52];
        B_392 = rconst(index,206)*var[126];
        B_393 = rconst(index,206)*var[56];
        B_394 = rconst(index,207)*var[126];
        B_395 = rconst(index,207)*var[114];
        B_396 = rconst(index,208)*var[126];
        B_397 = rconst(index,208)*var[114];
        B_398 = rconst(index,209)*var[136];
        B_399 = rconst(index,209)*var[114];
        B_400 = 1e-10*var[126];
        B_401 = 1e-10*var[65];
        B_402 = rconst(index,211);
        B_403 = 3e-13*var[124];
        B_404 = 3e-13*var[81];
        B_405 = 5e-11*var[137];
        B_406 = 5e-11*var[46];
        B_407 = 3.3e-10*var[127];
        B_408 = 3.3e-10*var[114];
        B_409 = rconst(index,215)*var[129];
        B_410 = rconst(index,215)*var[114];
        B_411 = 4.4e-13*var[132];
        B_412 = 4.4e-13*var[114];
        B_414 = rconst(index,218);
        B_415 = rconst(index,219);
        B_416 = rconst(index,220);
        B_417 = rconst(index,221);
        B_418 = rconst(index,222);
        B_419 = rconst(index,223);
        B_420 = rconst(index,224);
        B_421 = rconst(index,225);
        B_422 = rconst(index,226);
        B_423 = rconst(index,227);
        B_424 = rconst(index,228);
        B_425 = rconst(index,229);
        B_426 = rconst(index,230);
        B_427 = rconst(index,231);
        B_428 = rconst(index,232);
        B_429 = rconst(index,233);
        B_431 = rconst(index,235);
        B_432 = rconst(index,236);
        B_433 = rconst(index,237);
        B_434 = rconst(index,238);
        B_435 = rconst(index,239);
        B_436 = rconst(index,240);
        B_437 = rconst(index,241);
        B_438 = rconst(index,242);
        B_439 = rconst(index,243);
        B_440 = rconst(index,244);
        B_441 = rconst(index,245);
        B_442 = rconst(index,246);
        B_443 = rconst(index,247);
        B_444 = rconst(index,248);
        B_445 = rconst(index,249);
        B_446 = rconst(index,250);
        B_447 = rconst(index,251);
        B_448 = rconst(index,252);
        B_449 = rconst(index,253);
        B_450 = rconst(index,254);
        B_451 = rconst(index,255);
        B_452 = rconst(index,256);
        B_453 = rconst(index,257);
        B_454 = rconst(index,258);
        B_455 = rconst(index,259);
        B_456 = rconst(index,260);
        B_457 = rconst(index,261);
        B_458 = rconst(index,262);
        B_459 = rconst(index,263);
        B_460 = rconst(index,264);
        B_461 = rconst(index,265);
        B_462 = rconst(index,266);
        B_463 = rconst(index,267);
        B_464 = rconst(index,268);
        B_465 = rconst(index,269);
        B_466 = rconst(index,270);
        B_467 = rconst(index,271);
        B_468 = rconst(index,272);
        B_469 = rconst(index,273);
        B_470 = rconst(index,274);
        B_471 = rconst(index,275);
        B_472 = rconst(index,276);
        B_473 = rconst(index,277);
        B_474 = rconst(index,278);
        B_475 = rconst(index,279);
        B_476 = rconst(index,280);
        B_477 = rconst(index,281);
        B_478 = rconst(index,282);
        B_479 = rconst(index,283);
        B_480 = rconst(index,284);
        B_481 = rconst(index,285)*var[128];
        B_482 = rconst(index,285)*var[83];
        B_483 = rconst(index,286);
        B_484 = rconst(index,287)*var[138];
        B_485 = rconst(index,287)*var[112];
        B_486 = rconst(index,288)*var[138];
        B_487 = rconst(index,288)*var[116];
        B_488 = rconst(index,289)*var[128];
        B_489 = rconst(index,289)*var[116];
        B_490 = rconst(index,290)*var[138];
        B_491 = rconst(index,290)*var[83];
        B_492 = rconst(index,291)*var[123];
        B_493 = rconst(index,291)*var[118];
        B_494 = rconst(index,292)*var[128];
        B_495 = rconst(index,292)*var[105];
        B_496 = rconst(index,293)*var[123];
        B_497 = rconst(index,293)*var[116];
        B_498 = rconst(index,294)*var[138];
        B_499 = rconst(index,294)*var[105];
        B_500 = rconst(index,295)*var[123];
        B_501 = rconst(index,295)*var[112];
        B_502 = rconst(index,296)*var[138];
        B_503 = rconst(index,296)*var[118];
        B_504 = rconst(index,297);
        B_505 = 2.3e-10*var[120];
        B_506 = 2.3e-10*var[15];
        B_507 = rconst(index,299);
        B_508 = 1.4e-10*var[120];
        B_509 = 1.4e-10*var[16];
        B_510 = rconst(index,301);
        B_511 = rconst(index,302)*var[120];
        B_512 = rconst(index,302)*var[17];
        B_513 = rconst(index,303)*var[120];
        B_514 = rconst(index,303)*var[17];
        B_515 = rconst(index,304);
        B_516 = 3e-10*var[120];
        B_517 = 3e-10*var[18];
        B_518 = rconst(index,306)*var[126];
        B_519 = rconst(index,306)*var[18];
        B_520 = rconst(index,307);
        B_521 = rconst(index,308);
        B_522 = rconst(index,309);
        jcb[0] = - B_469;
        jcb[1] = - B_476;
        jcb[2] = - B_474;
        jcb[3] = - B_480;
        jcb[4] = - B_504;
        jcb[5] = - B_521;
        jcb[6] = - B_522;
        jcb[7] = B_476;
        jcb[8] = B_474;
        jcb[9] = 0;
        jcb[10] = B_313+ B_462;
        jcb[11] = B_327+ B_465;
        jcb[12] = B_329+ B_464;
        jcb[13] = B_370+ B_472;
        jcb[14] = B_368+ B_473;
        jcb[15] = B_390+ B_477;
        jcb[16] = B_362;
        jcb[17] = B_386+ B_478;
        jcb[18] = B_388+ B_479;
        jcb[19] = 2*B_319+ 2*B_321+ 2*B_463;
        jcb[20] = 0.9*B_315+ B_317;
        jcb[21] = B_314+ 0.9*B_316+ 2*B_320+ B_328+ B_330;
        jcb[22] = B_318+ 2*B_322+ B_363+ B_369+ B_371+ B_387+ B_389+ B_391;
        jcb[23] = 2*B_476;
        jcb[24] = 3*B_474;
        jcb[25] = 0;
        jcb[26] = 2*B_327+ 2*B_465;
        jcb[27] = B_329+ B_464;
        jcb[28] = 2*B_328+ B_330;
        jcb[29] = 0;
        jcb[30] = B_465;
        jcb[31] = 2*B_464;
        jcb[32] = B_390;
        jcb[33] = 2*B_386;
        jcb[34] = B_388;
        jcb[35] = 0.09*B_315;
        jcb[36] = 0.09*B_316;
        jcb[37] = 2*B_387+ B_389+ B_391;
        jcb[38] = 0;
        jcb[39] = B_405;
        jcb[40] = 0.4*B_400;
        jcb[41] = 0.4*B_401;
        jcb[42] = B_406;
        jcb[43] = 0;
        jcb[44] = B_392;
        jcb[45] = B_393;
        jcb[46] = 0;
        jcb[47] = 2*B_483;
        jcb[48] = 0;
        jcb[49] = 2*B_483;
        jcb[50] = B_521;
        jcb[51] = B_522;
        jcb[52] = 0;
        jcb[53] = B_507;
        jcb[54] = B_510;
        jcb[55] = B_513+ B_515;
        jcb[56] = B_520;
        jcb[57] = B_514;
        jcb[58] = - B_505- B_507;
        jcb[59] = - B_506;
        jcb[60] = - B_508- B_510;
        jcb[61] = - B_509;
        jcb[62] = - B_511- B_513- B_515;
        jcb[63] = - B_512- B_514;
        jcb[64] = - B_516- B_518- B_520;
        jcb[65] = - B_517;
        jcb[66] = - B_519;
        jcb[67] = B_504;
        jcb[68] = 0;
        jcb[69] = 0;
        jcb[70] = B_22;
        jcb[71] = B_23;
        jcb[72] = 0;
        jcb[73] = B_33;
        jcb[74] = B_34;
        jcb[75] = 0;
        jcb[76] = 2*B_454;
        jcb[77] = B_319;
        jcb[78] = B_41+ B_43;
        jcb[79] = B_315;
        jcb[80] = B_481+ 3*B_483+ 2*B_490;
        jcb[81] = B_93;
        jcb[82] = B_100;
        jcb[83] = B_79+ B_89+ B_91;
        jcb[84] = B_12;
        jcb[85] = B_108;
        jcb[86] = B_134;
        jcb[87] = B_498;
        jcb[88] = B_254+ B_258;
        jcb[89] = B_180+ B_184;
        jcb[90] = B_226;
        jcb[91] = B_457+ B_484+ B_500;
        jcb[92] = B_486+ B_496;
        jcb[93] = B_142;
        jcb[94] = B_343+ B_468+ B_492+ B_502;
        jcb[95] = B_150;
        jcb[96] = 2*B_4+ B_13+ B_33+ B_42+ B_44+ B_109+ B_316+ B_320;
        jcb[97] = B_162;
        jcb[98] = B_10;
        jcb[99] = B_493+ B_497+ B_501;
        jcb[100] = 2*B_5+ 2*B_6+ B_11+ B_16+ B_22+ B_80+ B_94+ B_135+ B_181+ B_227+ B_255;
        jcb[101] = B_118+ B_311+ B_360;
        jcb[102] = B_14+ B_17+ B_288;
        jcb[103] = B_34+ B_482;
        jcb[104] = B_126;
        jcb[105] = 2*B_7+ B_15+ B_20+ 2*B_49+ 2*B_273+ 2*B_333+ B_344;
        jcb[106] = 2*B_334+ 2*B_335+ 2*B_336+ B_361+ B_378+ 2*B_380+ 2*B_382;
        jcb[107] = 2*B_274+ 2*B_275+ 2*B_276+ B_277+ B_289+ B_312+ B_379+ 2*B_381+ 2*B_383;
        jcb[108] = 2*B_50+ B_90+ B_92+ B_101;
        jcb[109] = B_68+ B_119+ B_127+ B_143+ B_151+ B_163+ B_185+ B_259+ 2*B_422;
        jcb[110] = B_21+ B_23+ B_69;
        jcb[111] = B_485+ B_487+ 2*B_491+ B_499+ B_503;
        jcb[112] = 0;
        jcb[113] = 0.333333*B_498;
        jcb[114] = 0.5*B_500;
        jcb[115] = 0.333333*B_496;
        jcb[116] = B_343+ B_468+ B_492+ 0.5*B_502;
        jcb[117] = B_493+ 0.333333*B_497+ 0.5*B_501;
        jcb[118] = B_360;
        jcb[119] = 2*B_333+ B_344;
        jcb[120] = 2*B_334+ 2*B_335+ 2*B_336+ B_361+ 0.5*B_378+ B_380+ B_382;
        jcb[121] = 0.5*B_379+ B_381+ B_383;
        jcb[122] = 0.333333*B_499+ 0.5*B_503;
        jcb[123] = 0;
        jcb[124] = 2*B_454;
        jcb[125] = B_319;
        jcb[126] = B_315;
        jcb[127] = B_490;
        jcb[128] = 0.333333*B_498;
        jcb[129] = B_457+ B_484+ 0.5*B_500;
        jcb[130] = 0.5*B_486+ 0.333333*B_496;
        jcb[131] = 0.5*B_502;
        jcb[132] = B_316+ B_320;
        jcb[133] = 0.333333*B_497+ 0.5*B_501;
        jcb[134] = B_311;
        jcb[135] = B_288;
        jcb[136] = 2*B_273;
        jcb[137] = 0.5*B_378+ B_380+ B_382;
        jcb[138] = 2*B_274+ 2*B_275+ 2*B_276+ B_277+ B_289+ B_312+ 0.5*B_379+ B_381+ B_383;
        jcb[139] = B_485+ 0.5*B_487+ B_491+ 0.333333*B_499+ 0.5*B_503;
        jcb[140] = 0;
        jcb[141] = B_12;
        jcb[142] = B_13;
        jcb[143] = B_10;
        jcb[144] = B_11+ B_16+ B_22;
        jcb[145] = B_14+ B_17;
        jcb[146] = B_15+ B_20;
        jcb[147] = B_21+ B_23;
        jcb[148] = 0;
        jcb[149] = B_481+ 3*B_483+ B_490;
        jcb[150] = B_93;
        jcb[151] = B_100;
        jcb[152] = B_79+ B_89+ B_91;
        jcb[153] = 0.333333*B_498;
        jcb[154] = 0.5*B_486+ 0.333333*B_496;
        jcb[155] = 0.333333*B_497;
        jcb[156] = B_80+ B_94;
        jcb[157] = B_482;
        jcb[158] = 2*B_49;
        jcb[159] = 2*B_50+ B_90+ B_92+ B_101;
        jcb[160] = B_68+ 2*B_422;
        jcb[161] = B_69;
        jcb[162] = 0.5*B_487+ B_491+ 0.333333*B_499;
        jcb[163] = 0;
        jcb[164] = B_41+ B_43;
        jcb[165] = B_108;
        jcb[166] = 2*B_4+ B_33+ B_42+ B_44+ B_109;
        jcb[167] = 2*B_5+ 2*B_6;
        jcb[168] = B_34;
        jcb[169] = 2*B_7;
        jcb[170] = 0;
        jcb[171] = B_134;
        jcb[172] = B_254+ B_258;
        jcb[173] = B_180+ B_184;
        jcb[174] = B_226;
        jcb[175] = B_142;
        jcb[176] = B_150;
        jcb[177] = B_162;
        jcb[178] = B_135+ B_181+ B_227+ B_255;
        jcb[179] = B_118;
        jcb[180] = B_126;
        jcb[181] = B_119+ B_127+ B_143+ B_151+ B_163+ B_185+ B_259;
        jcb[182] = 0;
        jcb[183] = B_16;
        jcb[184] = B_17;
        jcb[185] = 0;
        jcb[186] = B_62;
        jcb[187] = B_63;
        jcb[188] = B_476;
        jcb[189] = B_474;
        jcb[190] = 0;
        jcb[191] = B_362+ B_471;
        jcb[192] = B_363;
        jcb[193] = B_476;
        jcb[194] = 0;
        jcb[195] = 4*B_313+ 4*B_462;
        jcb[196] = 2*B_327+ 2*B_465;
        jcb[197] = 3*B_329+ 3*B_464;
        jcb[198] = 3*B_319+ 3*B_321+ 3*B_463;
        jcb[199] = B_315+ B_317+ B_461;
        jcb[200] = 4*B_314+ B_316+ 3*B_320+ 2*B_328+ 3*B_330;
        jcb[201] = B_318+ 3*B_322;
        jcb[202] = 0;
        jcb[203] = B_116;
        jcb[204] = B_117;
        jcb[205] = B_469;
        jcb[206] = 0;
        jcb[207] = B_458;
        jcb[208] = B_455;
        jcb[209] = B_37+ B_47;
        jcb[210] = B_418;
        jcb[211] = 0.4*B_400;
        jcb[212] = 0.333*B_426;
        jcb[213] = B_70;
        jcb[214] = B_188;
        jcb[215] = B_204;
        jcb[216] = B_245;
        jcb[217] = B_345;
        jcb[218] = B_72;
        jcb[219] = B_222;
        jcb[220] = B_262;
        jcb[221] = B_232;
        jcb[222] = B_394+ B_396+ B_407+ B_409;
        jcb[223] = B_196;
        jcb[224] = B_140;
        jcb[225] = B_156+ B_158;
        jcb[226] = B_28;
        jcb[227] = B_116;
        jcb[228] = B_71+ B_73+ B_346+ B_395+ B_397+ 0.4*B_401;
        jcb[229] = B_284+ B_408;
        jcb[230] = B_410;
        jcb[231] = B_48+ B_62+ B_117+ B_141+ B_159+ B_189+ B_197+ B_205+ B_223+ B_233+ B_246+ B_263+ B_420;
        jcb[232] = B_29+ B_63+ B_157+ B_285;
        jcb[233] = 0;
        jcb[234] = B_188;
        jcb[235] = B_204;
        jcb[236] = B_245;
        jcb[237] = B_222;
        jcb[238] = B_262;
        jcb[239] = B_232;
        jcb[240] = B_196;
        jcb[241] = B_140;
        jcb[242] = B_158;
        jcb[243] = B_141+ B_159+ B_189+ B_197+ B_205+ B_223+ B_233+ B_246+ B_263;
        jcb[244] = 0;
        jcb[245] = 2*B_370+ 2*B_472;
        jcb[246] = 3*B_368+ 3*B_473;
        jcb[247] = B_390+ B_477;
        jcb[248] = B_386+ B_478;
        jcb[249] = 2*B_388+ 2*B_479;
        jcb[250] = 3*B_369+ 2*B_371+ B_387+ 2*B_389+ B_391;
        jcb[251] = 0;
        jcb[252] = B_477;
        jcb[253] = 2*B_478;
        jcb[254] = B_479;
        jcb[255] = - B_448;
        jcb[256] = 0.8*B_247;
        jcb[257] = 0.8*B_248;
        jcb[258] = - B_279- B_454;
        jcb[259] = B_278;
        jcb[260] = - B_216;
        jcb[261] = - B_217;
        jcb[262] = - B_313- B_462;
        jcb[263] = - B_314;
        jcb[264] = - B_327- B_465;
        jcb[265] = - B_328;
        jcb[266] = - B_329- B_464;
        jcb[267] = - B_330;
        jcb[268] = - B_370- B_472;
        jcb[269] = - B_371;
        jcb[270] = - B_368- B_473;
        jcb[271] = - B_369;
        jcb[272] = - B_405;
        jcb[273] = B_403;
        jcb[274] = B_404;
        jcb[275] = - B_406;
        jcb[276] = - B_77;
        jcb[277] = - B_78;
        jcb[278] = - B_132;
        jcb[279] = - B_133;
        jcb[280] = - B_178;
        jcb[281] = - B_179;
        jcb[282] = - B_458;
        jcb[283] = B_490;
        jcb[284] = B_491;
        jcb[285] = - B_455;
        jcb[286] = B_378;
        jcb[287] = B_277+ B_379;
        jcb[288] = - B_390- B_477;
        jcb[289] = - B_391;
        jcb[290] = - B_362- B_471;
        jcb[291] = - B_363;
        jcb[292] = - B_386- B_478;
        jcb[293] = - B_387;
        jcb[294] = - B_388- B_479;
        jcb[295] = - B_389;
        jcb[296] = - B_392;
        jcb[297] = 0.6*B_400;
        jcb[298] = B_402;
        jcb[299] = - B_393+ 0.6*B_401;
        jcb[300] = - B_319- B_321- B_463;
        jcb[301] = - B_320;
        jcb[302] = - B_322;
        jcb[303] = - B_173- B_435;
        jcb[304] = B_269;
        jcb[305] = - B_174+ B_270;
        jcb[306] = - B_37- B_47- B_53;
        jcb[307] = - B_48+ B_420;
        jcb[308] = - B_54;
        jcb[309] = B_53;
        jcb[310] = - B_41- B_43- B_418;
        jcb[311] = B_89;
        jcb[312] = - B_42- B_44;
        jcb[313] = 0;
        jcb[314] = B_54+ B_90;
        jcb[315] = - B_104;
        jcb[316] = B_98;
        jcb[317] = B_99;
        jcb[318] = - B_105;
        jcb[319] = - B_214- B_442;
        jcb[320] = 0.04*B_188;
        jcb[321] = - B_215;
        jcb[322] = 0.04*B_189;
        jcb[323] = - B_171- B_434;
        jcb[324] = B_154;
        jcb[325] = - B_172;
        jcb[326] = B_155;
        jcb[327] = - B_251- B_253- B_450;
        jcb[328] = B_234;
        jcb[329] = - B_252;
        jcb[330] = B_235;
        jcb[331] = - B_400;
        jcb[332] = B_396+ B_411;
        jcb[333] = B_397- B_401;
        jcb[334] = B_412;
        jcb[335] = - B_267- B_451;
        jcb[336] = B_260;
        jcb[337] = - B_268;
        jcb[338] = B_261;
        jcb[339] = - B_198;
        jcb[340] = B_194;
        jcb[341] = - B_199;
        jcb[342] = B_195;
        jcb[343] = - B_247- B_447;
        jcb[344] = B_243;
        jcb[345] = - B_248;
        jcb[346] = B_244;
        jcb[347] = - B_192- B_437;
        jcb[348] = B_186;
        jcb[349] = - B_193;
        jcb[350] = B_187;
        jcb[351] = B_104;
        jcb[352] = - B_98- B_102;
        jcb[353] = B_95;
        jcb[354] = - B_99;
        jcb[355] = - B_103+ B_105;
        jcb[356] = - B_146- B_432;
        jcb[357] = B_138;
        jcb[358] = - B_147;
        jcb[359] = B_139;
        jcb[360] = - B_208- B_441;
        jcb[361] = B_202;
        jcb[362] = - B_209;
        jcb[363] = B_203;
        jcb[364] = - B_74- B_75- B_426;
        jcb[365] = - B_76;
        jcb[366] = B_66;
        jcb[367] = B_67;
        jcb[368] = - B_152;
        jcb[369] = 0.18*B_168;
        jcb[370] = B_156+ B_166+ 0.18*B_169;
        jcb[371] = B_167;
        jcb[372] = - B_153;
        jcb[373] = B_157;
        jcb[374] = - 0.9*B_315- B_317- B_461;
        jcb[375] = - 0.9*B_316;
        jcb[376] = - B_318;
        jcb[377] = - B_70- B_424;
        jcb[378] = B_100;
        jcb[379] = B_60- B_71;
        jcb[380] = B_61;
        jcb[381] = B_101;
        jcb[382] = - B_175- B_177- B_436;
        jcb[383] = B_160;
        jcb[384] = - B_176;
        jcb[385] = B_161;
        jcb[386] = - B_130;
        jcb[387] = 0.23125*B_134;
        jcb[388] = 0.28*B_254;
        jcb[389] = 0.22*B_180;
        jcb[390] = 0.45*B_226;
        jcb[391] = 0.23125*B_135+ 0.22*B_181+ 0.45*B_227+ 0.28*B_255;
        jcb[392] = - B_131;
        jcb[393] = - B_224- B_443;
        jcb[394] = B_220;
        jcb[395] = - B_225;
        jcb[396] = B_221;
        jcb[397] = - B_374- B_453;
        jcb[398] = B_384;
        jcb[399] = B_484;
        jcb[400] = B_303+ B_486;
        jcb[401] = B_304+ B_385;
        jcb[402] = - B_375;
        jcb[403] = B_275;
        jcb[404] = B_485+ B_487;
        jcb[405] = - B_402- B_403;
        jcb[406] = B_394+ B_398+ B_407+ B_409;
        jcb[407] = - B_404;
        jcb[408] = B_395;
        jcb[409] = B_408;
        jcb[410] = B_410;
        jcb[411] = B_399;
        jcb[412] = - B_239- B_445;
        jcb[413] = B_230;
        jcb[414] = - B_240;
        jcb[415] = B_231;
        jcb[416] = - B_59- B_423- B_481- B_483- B_490;
        jcb[417] = - B_482;
        jcb[418] = B_57;
        jcb[419] = B_58;
        jcb[420] = - B_491;
        jcb[421] = - B_93- B_95;
        jcb[422] = B_79+ B_81+ B_91;
        jcb[423] = B_80- B_94;
        jcb[424] = B_92;
        jcb[425] = B_82;
        jcb[426] = 0.85*B_224+ 0.67*B_443;
        jcb[427] = - B_241- B_446;
        jcb[428] = 0.88*B_218+ 0.56*B_222;
        jcb[429] = B_249+ 0.67*B_449;
        jcb[430] = 0.88*B_219;
        jcb[431] = 0.85*B_225- B_242+ B_250;
        jcb[432] = 0.56*B_223;
        jcb[433] = 0;
        jcb[434] = B_214+ B_442;
        jcb[435] = 0.7*B_192+ B_437;
        jcb[436] = - B_200- B_438;
        jcb[437] = 0.96*B_188+ B_190;
        jcb[438] = B_191;
        jcb[439] = 0.7*B_193- B_201+ B_215;
        jcb[440] = 0.96*B_189;
        jcb[441] = 0;
        jcb[442] = - B_98+ B_102;
        jcb[443] = 0;
        jcb[444] = - B_96- B_99- B_100- B_106;
        jcb[445] = B_83;
        jcb[446] = 0;
        jcb[447] = - B_97+ B_103;
        jcb[448] = - B_101;
        jcb[449] = B_84;
        jcb[450] = - B_35- B_286- B_417;
        jcb[451] = 0.13875*B_134;
        jcb[452] = 0.09*B_254;
        jcb[453] = 0.13875*B_135+ 0.09*B_255;
        jcb[454] = - B_36;
        jcb[455] = - B_287;
        jcb[456] = B_32;
        jcb[457] = - B_112;
        jcb[458] = 0.2*B_190;
        jcb[459] = 0.5*B_206;
        jcb[460] = 0.18*B_218;
        jcb[461] = 0.03*B_180;
        jcb[462] = 0.25*B_264;
        jcb[463] = 0.25*B_236;
        jcb[464] = 0.25*B_144;
        jcb[465] = 0.03*B_181;
        jcb[466] = B_121+ 0.25*B_145+ 0.2*B_191+ 0.5*B_207+ 0.18*B_219+ 0.25*B_237+ 0.25*B_265;
        jcb[467] = - B_113;
        jcb[468] = B_374;
        jcb[469] = - B_372- B_384- B_475;
        jcb[470] = B_376;
        jcb[471] = B_498;
        jcb[472] = B_500;
        jcb[473] = B_496;
        jcb[474] = B_502;
        jcb[475] = B_497+ B_501;
        jcb[476] = B_377- B_385;
        jcb[477] = - B_373+ B_375;
        jcb[478] = B_382;
        jcb[479] = B_383;
        jcb[480] = B_499+ B_503;
        jcb[481] = - B_269- B_452;
        jcb[482] = B_258;
        jcb[483] = 0.044*B_262;
        jcb[484] = - B_270;
        jcb[485] = 0.044*B_263;
        jcb[486] = B_259;
        jcb[487] = B_77;
        jcb[488] = B_93;
        jcb[489] = - B_79- B_81- B_83- B_85- B_87- B_89- B_91;
        jcb[490] = - B_80+ B_94;
        jcb[491] = B_78;
        jcb[492] = - B_86- B_88;
        jcb[493] = - B_90- B_92;
        jcb[494] = - B_82- B_84;
        jcb[495] = 0.82*B_178;
        jcb[496] = 0.3*B_192;
        jcb[497] = - B_186- B_188- B_190;
        jcb[498] = - B_191;
        jcb[499] = 0.82*B_179+ 0.3*B_193;
        jcb[500] = - B_189;
        jcb[501] = - B_187;
        jcb[502] = 0.3*B_208;
        jcb[503] = B_200;
        jcb[504] = 0;
        jcb[505] = - B_202- B_204- B_206;
        jcb[506] = - B_207;
        jcb[507] = B_201+ 0.3*B_209;
        jcb[508] = - B_205;
        jcb[509] = - B_203;
        jcb[510] = B_173+ B_435;
        jcb[511] = B_175;
        jcb[512] = 0.25*B_445;
        jcb[513] = 0;
        jcb[514] = - B_128;
        jcb[515] = B_212+ B_440;
        jcb[516] = B_431;
        jcb[517] = 0.63*B_134;
        jcb[518] = 0.14*B_254;
        jcb[519] = 0.31*B_180;
        jcb[520] = 0;
        jcb[521] = 0.22*B_226+ B_444;
        jcb[522] = 0.25*B_232+ 0.125*B_236+ 0.5*B_238;
        jcb[523] = B_433;
        jcb[524] = 0;
        jcb[525] = 0.63*B_135+ 0.31*B_181+ 0.22*B_227+ 0.14*B_255;
        jcb[526] = 0.125*B_237;
        jcb[527] = B_124- B_129+ B_174+ B_176+ B_213;
        jcb[528] = B_307;
        jcb[529] = B_354;
        jcb[530] = B_125+ B_126+ B_308+ B_355+ B_428+ B_429;
        jcb[531] = 0.25*B_233;
        jcb[532] = 0;
        jcb[533] = B_127;
        jcb[534] = 0;
        jcb[535] = 0.7*B_208;
        jcb[536] = 0.5*B_445;
        jcb[537] = 0.5*B_206;
        jcb[538] = - B_212- B_440;
        jcb[539] = 0.04*B_180;
        jcb[540] = B_210;
        jcb[541] = 0.25*B_264;
        jcb[542] = 0.9*B_226;
        jcb[543] = 0.5*B_232+ 0.5*B_236+ B_238;
        jcb[544] = 0.04*B_181+ 0.9*B_227;
        jcb[545] = 0.5*B_207+ 0.5*B_237+ 0.25*B_265;
        jcb[546] = 0.7*B_209+ B_211- B_213;
        jcb[547] = 0.5*B_233;
        jcb[548] = 0;
        jcb[549] = - B_12- B_18- B_280;
        jcb[550] = 0.05*B_108+ 0.69*B_431;
        jcb[551] = - B_13+ 0.05*B_109;
        jcb[552] = B_26;
        jcb[553] = - B_19;
        jcb[554] = - B_281;
        jcb[555] = B_428;
        jcb[556] = B_27;
        jcb[557] = - B_108- B_110- B_305- B_431;
        jcb[558] = 0.06*B_180;
        jcb[559] = - B_109;
        jcb[560] = 0.06*B_181;
        jcb[561] = - B_111;
        jcb[562] = - B_306;
        jcb[563] = 0.2*B_247;
        jcb[564] = B_241;
        jcb[565] = - B_243- B_245;
        jcb[566] = 0;
        jcb[567] = 0;
        jcb[568] = 0;
        jcb[569] = B_242+ 0.2*B_248;
        jcb[570] = - B_246;
        jcb[571] = - B_244;
        jcb[572] = B_372;
        jcb[573] = - B_345- B_376- B_466;
        jcb[574] = B_347;
        jcb[575] = 0;
        jcb[576] = 0;
        jcb[577] = B_492;
        jcb[578] = B_493;
        jcb[579] = - B_346;
        jcb[580] = - B_377;
        jcb[581] = B_348+ B_373;
        jcb[582] = B_336;
        jcb[583] = 0;
        jcb[584] = 0;
        jcb[585] = 2*B_481+ B_490;
        jcb[586] = - B_72- B_425;
        jcb[587] = B_494+ B_498;
        jcb[588] = B_398;
        jcb[589] = B_486+ B_488+ B_496;
        jcb[590] = B_150;
        jcb[591] = B_497;
        jcb[592] = B_64- B_73;
        jcb[593] = 2*B_482+ B_489+ B_495;
        jcb[594] = B_126;
        jcb[595] = B_65;
        jcb[596] = B_127+ B_151+ B_399;
        jcb[597] = B_487+ B_491+ B_499;
        jcb[598] = B_216;
        jcb[599] = 0.15*B_224;
        jcb[600] = - B_218- B_220- B_222;
        jcb[601] = - B_219;
        jcb[602] = B_217+ 0.15*B_225;
        jcb[603] = - B_223;
        jcb[604] = - B_221;
        jcb[605] = - B_134- B_136- B_323- B_364;
        jcb[606] = - B_135;
        jcb[607] = - B_137;
        jcb[608] = - B_324;
        jcb[609] = - B_365;
        jcb[610] = - B_122- B_309- B_356- B_427;
        jcb[611] = B_114;
        jcb[612] = - B_123;
        jcb[613] = - B_310;
        jcb[614] = - B_357;
        jcb[615] = B_115;
        jcb[616] = - B_347- B_353- B_470- B_494- B_498;
        jcb[617] = - B_495;
        jcb[618] = - B_348;
        jcb[619] = B_351;
        jcb[620] = B_352;
        jcb[621] = - B_499;
        jcb[622] = - B_254- B_256- B_258;
        jcb[623] = - B_255;
        jcb[624] = - B_257;
        jcb[625] = - B_259;
        jcb[626] = - B_180- B_182- B_184;
        jcb[627] = - B_181;
        jcb[628] = - B_183;
        jcb[629] = - B_185;
        jcb[630] = B_251+ B_450;
        jcb[631] = 0.5*B_198;
        jcb[632] = 0.25*B_445;
        jcb[633] = B_269;
        jcb[634] = 0.2*B_206;
        jcb[635] = 0;
        jcb[636] = - B_210- B_439;
        jcb[637] = 0.25*B_264;
        jcb[638] = 0.25*B_232+ 0.375*B_236+ B_238;
        jcb[639] = 0;
        jcb[640] = 0;
        jcb[641] = 0.2*B_207+ 0.375*B_237+ 0.25*B_265;
        jcb[642] = 0.5*B_199- B_211+ B_252+ B_270;
        jcb[643] = 0.25*B_233;
        jcb[644] = 0;
        jcb[645] = 0;
        jcb[646] = 0;
        jcb[647] = B_256;
        jcb[648] = - B_260- B_262- B_264- 2*B_266;
        jcb[649] = 0;
        jcb[650] = - B_265;
        jcb[651] = B_257;
        jcb[652] = - B_263;
        jcb[653] = 0;
        jcb[654] = - B_261;
        jcb[655] = B_267+ B_451;
        jcb[656] = B_452;
        jcb[657] = 0.65*B_254;
        jcb[658] = 0.956*B_262+ 0.5*B_264+ 2*B_266;
        jcb[659] = - B_226- B_228- B_444;
        jcb[660] = - B_227+ 0.65*B_255;
        jcb[661] = 0.5*B_265;
        jcb[662] = - B_229+ B_268;
        jcb[663] = 0.956*B_263;
        jcb[664] = 0;
        jcb[665] = 0;
        jcb[666] = 0.015*B_245;
        jcb[667] = 0.16*B_222;
        jcb[668] = B_184;
        jcb[669] = - B_249- B_449;
        jcb[670] = 0.02*B_196;
        jcb[671] = 0;
        jcb[672] = 0;
        jcb[673] = - B_250;
        jcb[674] = 0.02*B_197+ 0.16*B_223+ 0.015*B_246;
        jcb[675] = B_185;
        jcb[676] = 0;
        jcb[677] = - B_294- B_457- B_484- B_500;
        jcb[678] = B_488;
        jcb[679] = - B_501;
        jcb[680] = - B_295;
        jcb[681] = B_489;
        jcb[682] = B_290;
        jcb[683] = B_291;
        jcb[684] = - B_485;
        jcb[685] = B_253;
        jcb[686] = B_239;
        jcb[687] = 0.1*B_254;
        jcb[688] = B_228;
        jcb[689] = - B_230- B_232- B_234- B_236- 2*B_238;
        jcb[690] = 0.1*B_255;
        jcb[691] = - B_237;
        jcb[692] = B_229+ B_240;
        jcb[693] = - B_233;
        jcb[694] = - B_235;
        jcb[695] = 0;
        jcb[696] = - B_231;
        jcb[697] = - B_394- B_396- B_398- B_407- B_409- B_411;
        jcb[698] = - B_395- B_397;
        jcb[699] = - B_408;
        jcb[700] = - B_410;
        jcb[701] = - B_412;
        jcb[702] = - B_399;
        jcb[703] = 0.5*B_198;
        jcb[704] = 0.666667*B_136+ 0.666667*B_323+ 0.666667*B_364;
        jcb[705] = B_182;
        jcb[706] = - B_194- B_196;
        jcb[707] = 0;
        jcb[708] = 0.666667*B_137+ B_183+ 0.5*B_199;
        jcb[709] = 0.666667*B_324;
        jcb[710] = 0.666667*B_365;
        jcb[711] = - B_197;
        jcb[712] = 0;
        jcb[713] = - B_195;
        jcb[714] = - B_300- B_301- B_303- B_459- B_460- B_486- B_488- B_496;
        jcb[715] = - B_497;
        jcb[716] = - B_304;
        jcb[717] = - B_489;
        jcb[718] = - B_302;
        jcb[719] = B_298;
        jcb[720] = B_299;
        jcb[721] = - B_487;
        jcb[722] = B_132;
        jcb[723] = 0.18*B_178;
        jcb[724] = 0.3*B_146;
        jcb[725] = 0.33*B_443;
        jcb[726] = B_446;
        jcb[727] = 0.12*B_218+ 0.28*B_222;
        jcb[728] = 0.06*B_180;
        jcb[729] = 0.33*B_449;
        jcb[730] = 0;
        jcb[731] = - B_138- B_140- B_142- B_144- B_168;
        jcb[732] = - B_169;
        jcb[733] = 0.06*B_181;
        jcb[734] = - B_145+ 0.12*B_219;
        jcb[735] = B_133+ 0.3*B_147+ 0.18*B_179;
        jcb[736] = 0;
        jcb[737] = 0;
        jcb[738] = - B_141+ 0.28*B_223;
        jcb[739] = - B_143;
        jcb[740] = - B_139;
        jcb[741] = B_345;
        jcb[742] = B_494;
        jcb[743] = 0;
        jcb[744] = 0;
        jcb[745] = - B_343- B_468- B_492- B_502;
        jcb[746] = - B_493;
        jcb[747] = B_358;
        jcb[748] = B_346;
        jcb[749] = 0;
        jcb[750] = B_495;
        jcb[751] = 0;
        jcb[752] = - B_344;
        jcb[753] = B_339+ B_359;
        jcb[754] = 0;
        jcb[755] = 0;
        jcb[756] = B_340;
        jcb[757] = - B_503;
        jcb[758] = B_447;
        jcb[759] = 0.7*B_146+ B_432;
        jcb[760] = 0.33*B_443;
        jcb[761] = 0.985*B_245;
        jcb[762] = 0.12*B_218+ 0.28*B_222;
        jcb[763] = 0.47*B_180;
        jcb[764] = 0.33*B_449;
        jcb[765] = 0.98*B_196;
        jcb[766] = B_140+ B_142+ 0.75*B_144+ B_168;
        jcb[767] = - B_148- B_150- B_325- B_366- B_433;
        jcb[768] = B_169;
        jcb[769] = 0.47*B_181;
        jcb[770] = 0.75*B_145+ 0.12*B_219;
        jcb[771] = 0.7*B_147- B_149;
        jcb[772] = - B_326;
        jcb[773] = - B_367;
        jcb[774] = B_141+ 0.98*B_197+ 0.28*B_223+ 0.985*B_246;
        jcb[775] = B_143- B_151;
        jcb[776] = 0;
        jcb[777] = - B_313;
        jcb[778] = - B_327;
        jcb[779] = - B_329;
        jcb[780] = - B_319;
        jcb[781] = - B_41- B_43+ B_418;
        jcb[782] = - B_315;
        jcb[783] = 0;
        jcb[784] = - B_12;
        jcb[785] = - B_108;
        jcb[786] = 0;
        jcb[787] = - B_0- B_4- B_13- B_33- B_39- B_42- B_44- B_109- B_314- B_316- B_320- B_328- B_330;
        jcb[788] = 0;
        jcb[789] = - B_5+ B_414;
        jcb[790] = 0;
        jcb[791] = 0;
        jcb[792] = - B_34;
        jcb[793] = 0;
        jcb[794] = 0;
        jcb[795] = 0;
        jcb[796] = 0;
        jcb[797] = 0;
        jcb[798] = 2*B_448;
        jcb[799] = B_171;
        jcb[800] = B_447;
        jcb[801] = B_441;
        jcb[802] = B_177+ B_436;
        jcb[803] = 0.25*B_445;
        jcb[804] = B_446;
        jcb[805] = B_438;
        jcb[806] = 0;
        jcb[807] = B_204+ 0.3*B_206;
        jcb[808] = B_212+ B_440;
        jcb[809] = 0.985*B_245;
        jcb[810] = 0;
        jcb[811] = 0.1*B_254;
        jcb[812] = 0.23*B_180;
        jcb[813] = B_439;
        jcb[814] = 0;
        jcb[815] = 0.1*B_226+ B_444;
        jcb[816] = 0;
        jcb[817] = 0.25*B_232+ 0.125*B_236;
        jcb[818] = 0;
        jcb[819] = - B_168;
        jcb[820] = B_148+ B_150+ B_325+ B_366;
        jcb[821] = - B_154- B_156- B_158- B_160- B_162- B_164- B_166- B_169- 2*B_170;
        jcb[822] = 0.23*B_181+ 0.1*B_227+ 0.1*B_255;
        jcb[823] = - B_165- B_167+ 0.3*B_207+ 0.125*B_237;
        jcb[824] = B_149+ B_172+ B_213;
        jcb[825] = B_326;
        jcb[826] = B_367;
        jcb[827] = - B_159+ B_205+ 0.25*B_233+ 0.985*B_246;
        jcb[828] = - B_161;
        jcb[829] = B_151- B_163;
        jcb[830] = - B_155- B_157;
        jcb[831] = 0.09*B_315;
        jcb[832] = B_128;
        jcb[833] = 0;
        jcb[834] = B_12+ B_18+ B_280;
        jcb[835] = 0.4*B_108+ 0.31*B_431;
        jcb[836] = 0;
        jcb[837] = 0;
        jcb[838] = 0;
        jcb[839] = 0;
        jcb[840] = 0;
        jcb[841] = 0;
        jcb[842] = 0;
        jcb[843] = 0;
        jcb[844] = 0;
        jcb[845] = B_13+ 0.4*B_109+ 0.09*B_316;
        jcb[846] = 0;
        jcb[847] = - B_8- B_10- B_24- B_26- B_28;
        jcb[848] = - B_11;
        jcb[849] = 0;
        jcb[850] = B_14+ B_19+ B_129;
        jcb[851] = B_281;
        jcb[852] = B_416;
        jcb[853] = 0;
        jcb[854] = B_429;
        jcb[855] = B_15;
        jcb[856] = 0;
        jcb[857] = 0;
        jcb[858] = 0;
        jcb[859] = - B_25- B_27- B_29;
        jcb[860] = B_456;
        jcb[861] = B_364;
        jcb[862] = B_356;
        jcb[863] = - B_500;
        jcb[864] = B_409;
        jcb[865] = - B_496;
        jcb[866] = - B_492;
        jcb[867] = B_366;
        jcb[868] = 0;
        jcb[869] = - B_341- B_493- B_497- B_501;
        jcb[870] = 0;
        jcb[871] = 0;
        jcb[872] = - B_342;
        jcb[873] = 0;
        jcb[874] = 0;
        jcb[875] = B_337+ B_354+ B_357+ B_365+ B_367+ B_410;
        jcb[876] = B_355;
        jcb[877] = 0;
        jcb[878] = 0;
        jcb[879] = 0;
        jcb[880] = 0;
        jcb[881] = 0;
        jcb[882] = 0;
        jcb[883] = B_338;
        jcb[884] = 0;
        jcb[885] = - B_403;
        jcb[886] = - B_93;
        jcb[887] = - B_79;
        jcb[888] = - B_134;
        jcb[889] = - B_254;
        jcb[890] = - B_180;
        jcb[891] = - B_226;
        jcb[892] = 0;
        jcb[893] = - B_4;
        jcb[894] = B_156;
        jcb[895] = - B_10;
        jcb[896] = - B_5- B_6- B_11- B_16- B_22- B_45- B_51- B_80- B_94- B_135- B_181- B_227- B_255- B_271- B_331- B_404 - B_414- B_415;
        jcb[897] = 0;
        jcb[898] = - B_17;
        jcb[899] = - B_272;
        jcb[900] = 0;
        jcb[901] = - B_332;
        jcb[902] = 0;
        jcb[903] = B_2- B_7;
        jcb[904] = 0;
        jcb[905] = - B_46;
        jcb[906] = - B_52;
        jcb[907] = 0;
        jcb[908] = - B_23+ B_157;
        jcb[909] = 0;
        jcb[910] = B_480;
        jcb[911] = B_471;
        jcb[912] = B_434;
        jcb[913] = 0.6*B_400;
        jcb[914] = B_152;
        jcb[915] = B_461;
        jcb[916] = B_402;
        jcb[917] = B_438;
        jcb[918] = - B_190;
        jcb[919] = - B_206;
        jcb[920] = 0.75*B_108+ B_110+ B_305;
        jcb[921] = - B_218;
        jcb[922] = 0.7*B_122+ B_356;
        jcb[923] = 0.08*B_254;
        jcb[924] = 0.07*B_180;
        jcb[925] = - B_264;
        jcb[926] = - B_236;
        jcb[927] = 0;
        jcb[928] = - B_144+ 0.82*B_168;
        jcb[929] = B_433;
        jcb[930] = 0.75*B_109;
        jcb[931] = B_158+ B_162- B_166+ 0.82*B_169+ 2*B_170;
        jcb[932] = 0;
        jcb[933] = 0.07*B_181+ 0.08*B_255;
        jcb[934] = - B_114- B_116- B_118- 2*B_120- 2*B_121- B_145- B_167- B_191- B_207- B_219- B_237- B_265- B_311- B_358 - B_360;
        jcb[935] = B_111+ 0.7*B_123+ B_153+ 0.6*B_401;
        jcb[936] = B_306;
        jcb[937] = 0;
        jcb[938] = B_357;
        jcb[939] = 0;
        jcb[940] = 0;
        jcb[941] = - B_359- B_361;
        jcb[942] = - B_117+ B_159;
        jcb[943] = - B_312;
        jcb[944] = 0;
        jcb[945] = - B_119+ B_163;
        jcb[946] = - B_115;
        jcb[947] = 0;
        jcb[948] = - B_216;
        jcb[949] = - B_370;
        jcb[950] = - B_368;
        jcb[951] = - B_77;
        jcb[952] = - B_132;
        jcb[953] = - B_178;
        jcb[954] = - B_390;
        jcb[955] = - B_362;
        jcb[956] = - B_386;
        jcb[957] = - B_388;
        jcb[958] = - B_392;
        jcb[959] = B_319- B_321;
        jcb[960] = - B_173;
        jcb[961] = - B_104;
        jcb[962] = - B_214;
        jcb[963] = - B_171+ B_434;
        jcb[964] = - B_251;
        jcb[965] = - B_400;
        jcb[966] = B_451;
        jcb[967] = - 0.5*B_198;
        jcb[968] = - 0.2*B_247+ B_447;
        jcb[969] = - 0.3*B_192+ B_437;
        jcb[970] = - B_102;
        jcb[971] = - 0.3*B_146+ B_432;
        jcb[972] = - 0.3*B_208+ B_441;
        jcb[973] = - B_75+ 0.333*B_426;
        jcb[974] = - B_152;
        jcb[975] = - B_317;
        jcb[976] = - B_70+ B_424;
        jcb[977] = - B_175;
        jcb[978] = - B_130;
        jcb[979] = - 0.15*B_224+ B_443;
        jcb[980] = 0;
        jcb[981] = - B_239+ B_445;
        jcb[982] = 0;
        jcb[983] = - B_241;
        jcb[984] = - B_200;
        jcb[985] = - B_96;
        jcb[986] = - B_35+ 2*B_417;
        jcb[987] = - B_112;
        jcb[988] = - B_269;
        jcb[989] = B_81+ B_85;
        jcb[990] = 0;
        jcb[991] = 0;
        jcb[992] = - B_128;
        jcb[993] = - B_212;
        jcb[994] = B_12- B_18;
        jcb[995] = 0.75*B_108- B_110;
        jcb[996] = 0;
        jcb[997] = - B_345;
        jcb[998] = - B_72+ B_425;
        jcb[999] = 0;
        jcb[1000] = 0.13*B_134- B_136;
        jcb[1001] = - 0.7*B_122+ B_309+ B_427;
        jcb[1002] = 0;
        jcb[1003] = 0.25*B_254- B_256;
        jcb[1004] = 0.33*B_180- B_182;
        jcb[1005] = - B_210;
        jcb[1006] = 0;
        jcb[1007] = 0.19*B_226- B_228;
        jcb[1008] = - B_249;
        jcb[1009] = - B_294+ B_457;
        jcb[1010] = 0;
        jcb[1011] = - B_394- B_396;
        jcb[1012] = 0;
        jcb[1013] = 0;
        jcb[1014] = 0;
        jcb[1015] = B_343+ B_468;
        jcb[1016] = - B_148;
        jcb[1017] = B_13+ 2*B_33+ 0.75*B_109+ B_320;
        jcb[1018] = 0;
        jcb[1019] = B_10+ 2*B_24;
        jcb[1020] = - B_341;
        jcb[1021] = B_11- B_16+ B_22+ 0.13*B_135+ 0.33*B_181+ 0.19*B_227+ 0.25*B_255;
        jcb[1022] = 0;
        jcb[1023] = - B_14- B_17- B_19- B_30- B_36- B_60- B_64- B_71- B_73- B_76- B_78- B_97- B_103- B_105- B_111- B_113- 0.7 *B_123- B_124- B_129- B_131- B_133- B_137- 0.3*B_147- B_149- B_153- B_172- B_174- B_176- B_179- B_183- 0.3 *B_193- 0.5*B_199- B_201- 0.3*B_209- B_211- B_213- B_215- B_217- 0.15*B_225- B_229- B_240- B_242- 0.2 *B_248- B_250- B_252- B_257- B_270- B_288- B_292- B_295- B_318- B_322- B_342- B_346- B_363- B_369- B_371 - B_387- B_389- B_391- B_393- B_395- B_397- B_401;
        jcb[1024] = B_284+ B_310;
        jcb[1025] = 2*B_34+ B_416;
        jcb[1026] = 0;
        jcb[1027] = - B_125;
        jcb[1028] = - B_15+ B_20+ B_344;
        jcb[1029] = 0;
        jcb[1030] = - B_61+ B_62+ B_86;
        jcb[1031] = - B_289;
        jcb[1032] = - B_65;
        jcb[1033] = B_68;
        jcb[1034] = B_21+ B_23+ 2*B_25- B_31+ B_63+ B_69+ B_82+ B_285;
        jcb[1035] = - B_293;
        jcb[1036] = B_476;
        jcb[1037] = 2*B_454;
        jcb[1038] = 3*B_313+ 4*B_462;
        jcb[1039] = B_327+ B_465;
        jcb[1040] = 2*B_329+ B_464;
        jcb[1041] = B_458;
        jcb[1042] = B_477;
        jcb[1043] = 2*B_478;
        jcb[1044] = B_479;
        jcb[1045] = 3*B_319+ 3*B_321+ 3*B_463;
        jcb[1046] = 0.35*B_315+ B_317+ B_461;
        jcb[1047] = B_374+ 2*B_453;
        jcb[1048] = 0;
        jcb[1049] = - B_286;
        jcb[1050] = B_372- B_384+ B_475;
        jcb[1051] = - B_280;
        jcb[1052] = - B_305;
        jcb[1053] = - B_376;
        jcb[1054] = - B_323;
        jcb[1055] = - B_309;
        jcb[1056] = 0;
        jcb[1057] = 0;
        jcb[1058] = 0;
        jcb[1059] = B_457;
        jcb[1060] = - B_407;
        jcb[1061] = - B_303+ B_459;
        jcb[1062] = 0;
        jcb[1063] = - B_325;
        jcb[1064] = 3*B_314+ 0.35*B_316+ 3*B_320+ B_328+ 2*B_330;
        jcb[1065] = 0;
        jcb[1066] = 0;
        jcb[1067] = 0;
        jcb[1068] = - B_271;
        jcb[1069] = B_311;
        jcb[1070] = 0.94*B_288+ B_292+ B_318+ 3*B_322;
        jcb[1071] = - B_272- B_281- B_282- B_284- B_287- B_304- B_306- B_307- B_310- B_324- B_326- B_377- B_385- B_408;
        jcb[1072] = 0;
        jcb[1073] = B_373+ B_375;
        jcb[1074] = - B_308;
        jcb[1075] = B_273;
        jcb[1076] = B_380;
        jcb[1077] = B_296;
        jcb[1078] = B_274+ 2*B_276+ B_277+ 0.94*B_289+ B_297+ B_312+ B_381;
        jcb[1079] = 0;
        jcb[1080] = 0;
        jcb[1081] = - B_283- B_285;
        jcb[1082] = B_293+ B_456;
        jcb[1083] = B_216;
        jcb[1084] = B_370;
        jcb[1085] = B_368;
        jcb[1086] = B_77;
        jcb[1087] = B_132;
        jcb[1088] = B_178;
        jcb[1089] = B_390;
        jcb[1090] = B_362;
        jcb[1091] = B_386;
        jcb[1092] = B_388;
        jcb[1093] = B_321;
        jcb[1094] = B_104;
        jcb[1095] = B_171;
        jcb[1096] = B_198;
        jcb[1097] = B_102;
        jcb[1098] = B_75;
        jcb[1099] = B_152;
        jcb[1100] = B_317;
        jcb[1101] = B_70;
        jcb[1102] = B_175;
        jcb[1103] = B_130;
        jcb[1104] = 0.85*B_224;
        jcb[1105] = - B_481;
        jcb[1106] = 0;
        jcb[1107] = B_200;
        jcb[1108] = B_96;
        jcb[1109] = B_35;
        jcb[1110] = B_83+ B_87+ B_89;
        jcb[1111] = 0;
        jcb[1112] = B_18;
        jcb[1113] = B_110+ 1.155*B_431;
        jcb[1114] = B_72;
        jcb[1115] = 0;
        jcb[1116] = 0;
        jcb[1117] = B_122;
        jcb[1118] = - B_494;
        jcb[1119] = 0;
        jcb[1120] = 0;
        jcb[1121] = 0;
        jcb[1122] = B_249;
        jcb[1123] = B_294+ B_484+ B_500;
        jcb[1124] = 0;
        jcb[1125] = 0;
        jcb[1126] = - B_488;
        jcb[1127] = 0;
        jcb[1128] = B_492+ B_502;
        jcb[1129] = B_148;
        jcb[1130] = - B_33;
        jcb[1131] = 0;
        jcb[1132] = B_28;
        jcb[1133] = B_341+ B_493+ B_501;
        jcb[1134] = 0;
        jcb[1135] = 0;
        jcb[1136] = B_19+ B_30+ B_36+ B_71+ B_73+ B_76+ B_78+ B_97+ B_103+ B_105+ B_111+ B_123+ B_124+ B_131+ B_133+ B_149 + B_153+ B_172+ B_176+ B_179+ B_199+ B_201+ B_217+ 0.85*B_225+ B_250+ B_292+ B_295+ B_318+ B_322+ B_342 + B_363+ B_369+ B_371+ B_387+ B_389+ B_391;
        jcb[1137] = 0;
        jcb[1138] = - B_34- B_416- B_482- B_489- B_495;
        jcb[1139] = 0;
        jcb[1140] = B_125;
        jcb[1141] = 0;
        jcb[1142] = 0;
        jcb[1143] = B_88;
        jcb[1144] = 0;
        jcb[1145] = B_90;
        jcb[1146] = 0;
        jcb[1147] = B_29+ B_31+ B_84;
        jcb[1148] = B_293+ B_485+ B_503;
        jcb[1149] = B_469;
        jcb[1150] = B_476;
        jcb[1151] = B_474;
        jcb[1152] = 2*B_370+ 2*B_472;
        jcb[1153] = 3*B_368+ 3*B_473;
        jcb[1154] = B_390+ B_477;
        jcb[1155] = B_362+ B_471;
        jcb[1156] = B_386+ B_478;
        jcb[1157] = 2*B_388+ 2*B_479;
        jcb[1158] = - B_374;
        jcb[1159] = - B_372+ B_384+ B_475;
        jcb[1160] = B_345+ B_376+ 2*B_466;
        jcb[1161] = - B_364;
        jcb[1162] = - B_356;
        jcb[1163] = - B_347+ 0.85*B_470;
        jcb[1164] = 0;
        jcb[1165] = - B_409+ B_411;
        jcb[1166] = 0;
        jcb[1167] = B_468;
        jcb[1168] = - B_366;
        jcb[1169] = 0;
        jcb[1170] = B_341;
        jcb[1171] = - B_331;
        jcb[1172] = B_360;
        jcb[1173] = B_342+ B_346+ B_363+ 3*B_369+ 2*B_371+ B_387+ 2*B_389+ B_391;
        jcb[1174] = B_377+ B_385;
        jcb[1175] = 0;
        jcb[1176] = - B_332- B_337- B_348- B_354- B_357- B_365- B_367- B_373- B_375- B_410;
        jcb[1177] = - B_355;
        jcb[1178] = B_333;
        jcb[1179] = B_334+ 2*B_335+ B_349+ B_361+ B_378+ B_380+ B_412+ B_467;
        jcb[1180] = B_350;
        jcb[1181] = B_379+ B_381;
        jcb[1182] = 0;
        jcb[1183] = 0;
        jcb[1184] = - B_338;
        jcb[1185] = 0;
        jcb[1186] = B_173+ B_435;
        jcb[1187] = B_400;
        jcb[1188] = B_451;
        jcb[1189] = B_441;
        jcb[1190] = B_175;
        jcb[1191] = 0.75*B_445;
        jcb[1192] = B_112;
        jcb[1193] = B_452;
        jcb[1194] = 0.8*B_190;
        jcb[1195] = B_204+ 0.8*B_206;
        jcb[1196] = 0.25*B_108;
        jcb[1197] = 0.68*B_218;
        jcb[1198] = 1.13875*B_134;
        jcb[1199] = 0.3*B_122+ B_309+ B_427;
        jcb[1200] = 0.58*B_254;
        jcb[1201] = 0.57*B_180;
        jcb[1202] = B_439;
        jcb[1203] = 0.956*B_262+ 1.25*B_264+ B_266;
        jcb[1204] = B_444;
        jcb[1205] = 0.75*B_232+ 1.125*B_236+ 0.5*B_238;
        jcb[1206] = B_394+ B_398+ B_407+ B_409;
        jcb[1207] = 0.98*B_196;
        jcb[1208] = 0.75*B_144;
        jcb[1209] = 0.25*B_109;
        jcb[1210] = B_164+ B_166;
        jcb[1211] = 0;
        jcb[1212] = 1.13875*B_135+ 0.57*B_181+ 0.58*B_255;
        jcb[1213] = B_116+ B_118+ 2*B_120+ B_121+ 0.75*B_145+ B_165+ B_167+ 0.8*B_191+ 0.8*B_207+ 0.68*B_219+ 1.125*B_237  + 1.25*B_265+ B_311+ B_358+ B_360;
        jcb[1214] = B_113+ 0.3*B_123- B_124+ B_174+ B_176+ B_395+ B_401;
        jcb[1215] = - B_307+ B_310+ B_408;
        jcb[1216] = 0;
        jcb[1217] = - B_354+ B_410;
        jcb[1218] = - B_125- B_126- B_308- B_355- B_428- B_429;
        jcb[1219] = 0;
        jcb[1220] = B_359+ B_361;
        jcb[1221] = B_117+ 0.98*B_197+ B_205+ 0.75*B_233+ 0.956*B_263;
        jcb[1222] = B_312;
        jcb[1223] = 0;
        jcb[1224] = B_119- B_127+ B_399;
        jcb[1225] = 0;
        jcb[1226] = 0;
        jcb[1227] = B_455;
        jcb[1228] = B_37+ B_47+ B_53;
        jcb[1229] = 0.1*B_315;
        jcb[1230] = - B_301;
        jcb[1231] = - B_343;
        jcb[1232] = B_0+ B_39+ 0.1*B_316;
        jcb[1233] = B_28;
        jcb[1234] = 0;
        jcb[1235] = - B_6+ B_415;
        jcb[1236] = 0;
        jcb[1237] = - B_14;
        jcb[1238] = 0;
        jcb[1239] = 0;
        jcb[1240] = 0;
        jcb[1241] = 0;
        jcb[1242] = - B_2- B_7- B_15- B_20- B_49- B_273- B_302- B_333- B_344;
        jcb[1243] = - B_334+ B_467;
        jcb[1244] = B_48+ B_420;
        jcb[1245] = - B_274;
        jcb[1246] = - B_50+ B_54+ B_419;
        jcb[1247] = B_421;
        jcb[1248] = - B_21+ B_29;
        jcb[1249] = 0;
        jcb[1250] = B_353+ 0.15*B_470;
        jcb[1251] = - B_411;
        jcb[1252] = B_343;
        jcb[1253] = 0;
        jcb[1254] = B_331;
        jcb[1255] = - B_358- B_360;
        jcb[1256] = 0;
        jcb[1257] = 0;
        jcb[1258] = 0;
        jcb[1259] = B_332;
        jcb[1260] = 0;
        jcb[1261] = - B_333+ B_344;
        jcb[1262] = - B_334- 2*B_335- 2*B_336- B_339- B_349- B_351- B_359- B_361- B_378- B_380- B_382- B_412- B_467;
        jcb[1263] = - B_350;
        jcb[1264] = - B_379- B_381- B_383;
        jcb[1265] = - B_352;
        jcb[1266] = 0;
        jcb[1267] = - B_340;
        jcb[1268] = 0;
        jcb[1269] = B_37- B_47;
        jcb[1270] = 2*B_41;
        jcb[1271] = B_98;
        jcb[1272] = B_424;
        jcb[1273] = 0;
        jcb[1274] = B_96+ B_99+ B_100+ B_106;
        jcb[1275] = - B_85- B_87+ B_91;
        jcb[1276] = - B_188;
        jcb[1277] = - B_204;
        jcb[1278] = - B_245;
        jcb[1279] = - B_222;
        jcb[1280] = - B_262;
        jcb[1281] = 0;
        jcb[1282] = - B_232;
        jcb[1283] = - B_196;
        jcb[1284] = - B_140;
        jcb[1285] = 2*B_42;
        jcb[1286] = - B_158;
        jcb[1287] = 0;
        jcb[1288] = - B_45;
        jcb[1289] = - B_116;
        jcb[1290] = - B_60+ B_97;
        jcb[1291] = 0;
        jcb[1292] = 0;
        jcb[1293] = 0;
        jcb[1294] = 0;
        jcb[1295] = B_49;
        jcb[1296] = - B_349;
        jcb[1297] = - B_46- B_48- B_55- B_61- B_62- B_86- B_88- B_117- B_141- B_159- B_189- B_197- B_205- B_223- B_233- B_246 - B_263- B_296- B_350- B_420;
        jcb[1298] = - B_297;
        jcb[1299] = B_50+ B_92+ B_101+ B_419;
        jcb[1300] = - B_56+ B_422;
        jcb[1301] = - B_63;
        jcb[1302] = 0;
        jcb[1303] = 2*B_279;
        jcb[1304] = B_313;
        jcb[1305] = B_327;
        jcb[1306] = B_329;
        jcb[1307] = B_455;
        jcb[1308] = 0.46*B_315;
        jcb[1309] = B_294;
        jcb[1310] = B_300+ B_301+ B_460;
        jcb[1311] = B_314+ 0.46*B_316+ B_328+ B_330;
        jcb[1312] = 0;
        jcb[1313] = 0;
        jcb[1314] = B_271;
        jcb[1315] = - B_311;
        jcb[1316] = - B_288+ B_295;
        jcb[1317] = B_272+ B_284;
        jcb[1318] = 0;
        jcb[1319] = 0;
        jcb[1320] = 0;
        jcb[1321] = - B_273+ B_302;
        jcb[1322] = - B_378- B_380- B_382;
        jcb[1323] = - B_296;
        jcb[1324] = - B_274- 2*B_275- 2*B_276- 2*B_277- 2*B_278- B_289- B_290- B_297- B_298- B_312- B_379- B_381- B_383;
        jcb[1325] = - B_299;
        jcb[1326] = 0;
        jcb[1327] = B_285- B_291;
        jcb[1328] = 0;
        jcb[1329] = B_469;
        jcb[1330] = B_458;
        jcb[1331] = B_173+ B_435;
        jcb[1332] = - B_53;
        jcb[1333] = B_214+ B_442;
        jcb[1334] = B_251+ B_253+ B_450;
        jcb[1335] = B_74+ B_75+ 0.667*B_426;
        jcb[1336] = B_70;
        jcb[1337] = B_175+ B_177+ B_436;
        jcb[1338] = B_59+ B_423;
        jcb[1339] = - B_100;
        jcb[1340] = B_452;
        jcb[1341] = - B_89- B_91;
        jcb[1342] = 0.96*B_188;
        jcb[1343] = B_204;
        jcb[1344] = 0.985*B_245;
        jcb[1345] = B_425;
        jcb[1346] = 0.84*B_222;
        jcb[1347] = B_353+ 0.15*B_470;
        jcb[1348] = 0;
        jcb[1349] = 0.956*B_262;
        jcb[1350] = B_249+ B_449;
        jcb[1351] = B_232- B_234;
        jcb[1352] = 0;
        jcb[1353] = 0.98*B_196;
        jcb[1354] = B_300+ B_460;
        jcb[1355] = B_140+ B_142;
        jcb[1356] = 0;
        jcb[1357] = B_158- B_160+ B_162;
        jcb[1358] = 0;
        jcb[1359] = B_45- B_51;
        jcb[1360] = B_116+ B_118;
        jcb[1361] = - B_64+ B_71+ B_76+ B_174+ B_176+ B_215+ B_250+ B_252;
        jcb[1362] = 0;
        jcb[1363] = 0;
        jcb[1364] = 0;
        jcb[1365] = 0;
        jcb[1366] = - B_49;
        jcb[1367] = B_349- B_351;
        jcb[1368] = B_46+ 2*B_55+ B_62+ B_117+ B_141+ B_159+ 0.96*B_189+ 0.98*B_197+ B_205+ 0.84*B_223+ B_233+ 0.985*B_246  + 0.956*B_263+ B_296+ B_350;
        jcb[1369] = B_297- B_298;
        jcb[1370] = - B_50- B_52- B_54- B_57- B_65- B_66- B_90- B_92- B_101- B_161- B_235- B_299- B_352- B_419;
        jcb[1371] = 2*B_56- B_58+ B_68+ B_119+ B_143+ B_163+ B_421;
        jcb[1372] = B_63- B_67+ B_69;
        jcb[1373] = 0;
        jcb[1374] = 0.333*B_426;
        jcb[1375] = B_59+ B_423;
        jcb[1376] = B_72;
        jcb[1377] = B_347+ 0.85*B_470;
        jcb[1378] = - B_258;
        jcb[1379] = - B_184;
        jcb[1380] = - B_398;
        jcb[1381] = B_301+ B_303+ B_459;
        jcb[1382] = - B_142;
        jcb[1383] = - B_150;
        jcb[1384] = - B_162;
        jcb[1385] = 0;
        jcb[1386] = B_51;
        jcb[1387] = - B_118;
        jcb[1388] = B_73;
        jcb[1389] = B_304;
        jcb[1390] = 0;
        jcb[1391] = B_348;
        jcb[1392] = - B_126;
        jcb[1393] = B_302;
        jcb[1394] = 0;
        jcb[1395] = - B_55;
        jcb[1396] = 0;
        jcb[1397] = B_52- B_57;
        jcb[1398] = - B_56- B_58- B_68- B_119- B_127- B_143- B_151- B_163- B_185- B_259- B_399- B_421- B_422;
        jcb[1399] = - B_69;
        jcb[1400] = 0;
        jcb[1401] = - B_405;
        jcb[1402] = B_392;
        jcb[1403] = B_442;
        jcb[1404] = 0.4*B_400;
        jcb[1405] = B_451;
        jcb[1406] = B_437;
        jcb[1407] = B_432;
        jcb[1408] = B_74+ 0.667*B_426;
        jcb[1409] = B_130;
        jcb[1410] = 0.67*B_443;
        jcb[1411] = 0;
        jcb[1412] = 0.75*B_445;
        jcb[1413] = B_106;
        jcb[1414] = B_35+ B_286;
        jcb[1415] = B_112;
        jcb[1416] = B_452;
        jcb[1417] = - B_81- B_83+ B_85;
        jcb[1418] = - B_186+ 0.96*B_188+ 0.8*B_190;
        jcb[1419] = - B_202+ 0.3*B_206;
        jcb[1420] = B_440;
        jcb[1421] = - B_243;
        jcb[1422] = 1.23*B_218- B_220+ 0.56*B_222;
        jcb[1423] = 0.13*B_134;
        jcb[1424] = B_427;
        jcb[1425] = 0.25*B_254;
        jcb[1426] = 0.26*B_180;
        jcb[1427] = B_210+ B_439;
        jcb[1428] = - B_260+ 0.956*B_262+ B_264+ B_266;
        jcb[1429] = 0.32*B_226+ B_444;
        jcb[1430] = 0.67*B_449;
        jcb[1431] = - B_230+ 0.75*B_232+ 0.875*B_236+ B_238;
        jcb[1432] = B_396;
        jcb[1433] = - B_194+ 0.98*B_196;
        jcb[1434] = - B_138+ B_140+ B_142+ B_144+ 0.82*B_168;
        jcb[1435] = B_433;
        jcb[1436] = - B_154- B_156+ B_164+ 0.82*B_169;
        jcb[1437] = B_8- B_24- B_26- B_28;
        jcb[1438] = B_16- B_22+ 0.13*B_135+ 0.26*B_181+ 0.32*B_227+ 0.25*B_255;
        jcb[1439] = - B_114+ B_116+ B_118+ 2*B_120+ B_145+ B_165+ 0.8*B_191+ 0.3*B_207+ 1.23*B_219+ 0.875*B_237+ B_265+ B_311 + B_360;
        jcb[1440] = B_17- B_30+ B_36+ B_113+ B_124+ B_131+ B_211+ 0.94*B_288+ B_393+ B_397+ 0.4*B_401;
        jcb[1441] = - B_282- B_284+ B_287+ B_307;
        jcb[1442] = 0;
        jcb[1443] = - B_337+ B_354;
        jcb[1444] = B_125+ B_126+ B_308+ B_355+ B_429;
        jcb[1445] = - B_20;
        jcb[1446] = - B_339+ B_361;
        jcb[1447] = - B_62+ B_86+ B_117+ B_141+ 0.96*B_189+ 0.98*B_197+ 0.56*B_223+ 0.75*B_233+ 0.956*B_263;
        jcb[1448] = 0.94*B_289- B_290+ B_312;
        jcb[1449] = - B_66;
        jcb[1450] = - B_68+ B_119+ B_127+ B_143;
        jcb[1451] = - B_21- B_23- B_25- B_27- B_29- B_31- 2*B_32- B_63- B_67- B_69- B_82- B_84- B_115- B_139- B_155- B_157 - B_187- B_195- B_203- B_221- B_231- B_244- B_261- B_283- B_285- B_291- B_338- B_340- B_406;
        jcb[1452] = 0;
        jcb[1453] = - B_490;
        jcb[1454] = B_286;
        jcb[1455] = B_280;
        jcb[1456] = B_305;
        jcb[1457] = B_323;
        jcb[1458] = B_309;
        jcb[1459] = - B_498;
        jcb[1460] = 0;
        jcb[1461] = 0;
        jcb[1462] = - B_484;
        jcb[1463] = B_407;
        jcb[1464] = - B_486;
        jcb[1465] = - B_502;
        jcb[1466] = B_325;
        jcb[1467] = 0;
        jcb[1468] = 0;
        jcb[1469] = 0;
        jcb[1470] = 0;
        jcb[1471] = 0;
        jcb[1472] = 0;
        jcb[1473] = 0.06*B_288- B_292;
        jcb[1474] = B_281+ B_282+ B_287+ B_306+ B_307+ B_310+ B_324+ B_326+ B_408;
        jcb[1475] = 0;
        jcb[1476] = 0;
        jcb[1477] = B_308;
        jcb[1478] = 0;
        jcb[1479] = 0;
        jcb[1480] = 0;
        jcb[1481] = 0.06*B_289;
        jcb[1482] = 0;
        jcb[1483] = 0;
        jcb[1484] = B_283;
        jcb[1485] = - B_293- B_456- B_485- B_487- B_491- B_499- B_503;
    }

__device__ void Fun(double *var, const double * __restrict__ fix, const double * __restrict__ rconst, double *varDot, int &Nfun, const int VL_GLO){
    int index = blockIdx.x*blockDim.x+threadIdx.x;

    Nfun++;

 double dummy, A_0, A_1, A_2, A_3, A_4, A_5, A_6, A_7, A_8, A_9, A_10, A_11, A_12, A_13, A_14, A_15, A_16, A_17, A_18, A_19, A_20, A_21, A_22, A_23, A_24, A_25, A_26, A_27, A_28, A_29, A_30, A_31, A_32, A_33, A_34, A_35, A_36, A_37, A_38, A_39, A_40, A_41, A_42, A_43, A_44, A_45, A_46, A_47, A_48, A_49, A_50, A_51, A_52, A_53, A_54, A_55, A_56, A_57, A_58, A_59, A_60, A_61, A_62, A_63, A_64, A_65, A_66, A_67, A_68, A_69, A_70, A_71, A_72, A_73, A_74, A_75, A_76, A_77, A_78, A_79, A_80, A_81, A_82, A_83, A_84, A_85, A_86, A_87, A_88, A_89, A_90, A_91, A_92, A_93, A_94, A_95, A_96, A_97, A_98, A_99, A_100, A_101, A_102, A_103, A_104, A_105, A_106, A_107, A_108, A_109, A_110, A_111, A_112, A_113, A_114, A_115, A_116, A_117, A_118, A_119, A_120, A_121, A_122, A_123, A_124, A_125, A_126, A_127, A_128, A_129, A_130, A_131, A_132, A_133, A_134, A_135, A_136, A_137, A_138, A_139, A_140, A_141, A_142, A_143, A_144, A_145, A_146, A_147, A_148, A_149, A_150, A_151, A_152, A_153, A_154, A_155, A_156, A_157, A_158, A_159, A_160, A_161, A_162, A_163, A_164, A_165, A_166, A_167, A_168, A_169, A_170, A_171, A_172, A_173, A_174, A_175, A_176, A_177, A_178, A_179, A_180, A_181, A_182, A_183, A_184, A_185, A_186, A_187, A_188, A_189, A_190, A_191, A_192, A_193, A_194, A_195, A_196, A_197, A_198, A_199, A_200, A_201, A_202, A_203, A_204, A_205, A_206, A_207, A_208, A_209, A_210, A_211, A_212, A_213, A_214, A_215, A_216, A_217, A_218, A_219, A_220, A_221, A_222, A_223, A_224, A_225, A_226, A_227, A_228, A_229, A_230, A_231, A_232, A_233, A_234, A_235, A_236, A_237, A_238, A_239, A_240, A_241, A_242, A_243, A_244, A_245, A_246, A_247, A_248, A_249, A_250, A_251, A_252, A_253, A_254, A_255, A_256, A_257, A_258, A_259, A_260, A_261, A_262, A_263, A_264, A_265, A_266, A_267, A_268, A_269, A_270, A_271, A_272, A_273, A_274, A_275, A_276, A_277, A_278, A_279, A_280, A_281, A_282, A_283, A_284, A_285, A_286, A_287, A_288, A_289, A_290, A_291, A_292, A_293, A_294, A_295, A_296, A_297, A_298, A_299, A_300, A_301, A_302, A_303, A_304, A_305, A_306, A_307, A_308, A_309;

    {
        A_0 = rconst(index,0)*var[120]*fix(index,0);
        A_1 = rconst(index,1)*var[131]*fix(index,0);
        A_2 = 1.2e-10*var[120]*var[124];
        A_3 = rconst(index,3)*var[124]*var[131];
        A_4 = rconst(index,4)*var[122]*fix(index,0);
        A_5 = rconst(index,5)*var[122]*var[124];
        A_6 = 1.2e-10*var[97]*var[120];
        A_7 = rconst(index,7)*var[126]*var[131];
        A_8 = rconst(index,8)*var[124]*var[126];
        A_9 = rconst(index,9)*var[97]*var[126];
        A_10 = rconst(index,10)*var[131]*var[137];
        A_11 = rconst(index,11)*var[124]*var[137];
        A_12 = 7.2e-11*var[122]*var[137];
        A_13 = 6.9e-12*var[122]*var[137];
        A_14 = 1.6e-12*var[122]*var[137];
        A_15 = rconst(index,15)*var[126]*var[137];
        A_16 = rconst(index,16)*var[137]*var[137];
        A_17 = rconst(index,17)*var[120]*var[128];
        A_18 = 1.8e-12*var[88]*var[126];
        A_19 = rconst(index,19)*var[59]*fix(index,0);
        A_20 = rconst(index,20)*var[120]*fix(index,1);
        A_21 = rconst(index,21)*var[60]*var[120];
        A_22 = rconst(index,22)*var[60]*var[120];
        A_23 = rconst(index,23)*var[124]*var[133];
        A_24 = rconst(index,24)*var[59]*var[133];
        A_25 = rconst(index,25)*var[131]*var[135];
        A_26 = rconst(index,26)*var[124]*var[135];
        A_27 = rconst(index,27)*var[59]*var[135];
        A_28 = rconst(index,28)*var[133]*var[136];
        A_29 = rconst(index,29)*var[135]*var[136];
        A_30 = rconst(index,30)*var[83];
        A_31 = rconst(index,31)*var[126]*var[133];
        A_32 = rconst(index,32)*var[133]*var[137];
        A_33 = rconst(index,33)*var[126]*var[135];
        A_34 = rconst(index,34)*var[135]*var[137];
        A_35 = 3.5e-12*var[136]*var[137];
        A_36 = rconst(index,36)*var[76]*var[126];
        A_37 = rconst(index,37)*var[101]*var[126];
        A_38 = rconst(index,38)*var[73];
        A_39 = rconst(index,39)*var[73]*var[126];
        A_40 = rconst(index,40)*var[47]*var[126];
        A_41 = rconst(index,41)*var[92]*var[124];
        A_42 = rconst(index,42)*var[92]*var[137];
        A_43 = rconst(index,43)*var[92]*var[137];
        A_44 = rconst(index,44)*var[92]*var[133];
        A_45 = rconst(index,45)*var[92]*var[133];
        A_46 = rconst(index,46)*var[92]*var[135];
        A_47 = rconst(index,47)*var[92]*var[135];
        A_48 = 1.2e-14*var[84]*var[124];
        A_49 = 1300*var[84];
        A_50 = rconst(index,50)*var[87]*var[126];
        A_51 = rconst(index,51)*var[70]*var[87];
        A_52 = rconst(index,52)*var[87]*var[135];
        A_53 = 1.66e-12*var[70]*var[126];
        A_54 = rconst(index,54)*var[61]*var[126];
        A_55 = rconst(index,55)*var[87]*fix(index,0);
        A_56 = 1.75e-10*var[98]*var[120];
        A_57 = rconst(index,57)*var[98]*var[126];
        A_58 = rconst(index,58)*var[89]*var[126];
        A_59 = rconst(index,59)*var[125]*var[137];
        A_60 = rconst(index,60)*var[125]*var[133];
        A_61 = 1.3e-12*var[125]*var[136];
        A_62 = rconst(index,62)*var[125]*var[125];
        A_63 = rconst(index,63)*var[125]*var[125];
        A_64 = rconst(index,64)*var[104]*var[126];
        A_65 = rconst(index,65)*var[126]*var[130];
        A_66 = rconst(index,66)*var[130]*var[136];
        A_67 = rconst(index,67)*var[95]*var[126];
        A_68 = 4e-13*var[78]*var[126];
        A_69 = rconst(index,69)*var[48]*var[126];
        A_70 = rconst(index,70)*var[103]*var[124];
        A_71 = rconst(index,71)*var[103]*var[126];
        A_72 = rconst(index,72)*var[117]*var[137];
        A_73 = rconst(index,73)*var[117]*var[133];
        A_74 = 2.3e-12*var[117]*var[136];
        A_75 = rconst(index,75)*var[117]*var[125];
        A_76 = rconst(index,76)*var[71]*var[126];
        A_77 = rconst(index,77)*var[119]*var[126];
        A_78 = rconst(index,78)*var[119]*var[136];
        A_79 = rconst(index,79)*var[74]*var[126];
        A_80 = rconst(index,80)*var[121]*var[137];
        A_81 = rconst(index,81)*var[121]*var[137];
        A_82 = rconst(index,82)*var[121]*var[133];
        A_83 = rconst(index,83)*var[121]*var[135];
        A_84 = 4e-12*var[121]*var[136];
        A_85 = rconst(index,85)*var[121]*var[125];
        A_86 = rconst(index,86)*var[121]*var[125];
        A_87 = rconst(index,87)*var[117]*var[121];
        A_88 = rconst(index,88)*var[121]*var[121];
        A_89 = rconst(index,89)*var[63]*var[126];
        A_90 = rconst(index,90)*var[58]*var[126];
        A_91 = rconst(index,91)*var[77]*var[126];
        A_92 = rconst(index,92)*var[77];
        A_93 = rconst(index,93)*var[49]*var[126];
        A_94 = rconst(index,94)*var[107]*var[124];
        A_95 = rconst(index,95)*var[107]*var[126];
        A_96 = rconst(index,96)*var[107]*var[136];
        A_97 = rconst(index,97)*var[93]*var[137];
        A_98 = rconst(index,98)*var[93]*var[133];
        A_99 = rconst(index,99)*var[93]*var[125];
        A_100 = rconst(index,100)*var[69]*var[126];
        A_101 = rconst(index,101)*var[115]*var[137];
        A_102 = rconst(index,102)*var[115]*var[133];
        A_103 = rconst(index,103)*var[67]*var[126];
        A_104 = rconst(index,104)*var[86]*var[126];
        A_105 = rconst(index,105)*var[94]*var[137];
        A_106 = rconst(index,106)*var[94]*var[133];
        A_107 = rconst(index,107)*var[94]*var[125];
        A_108 = rconst(index,108)*var[72]*var[126];
        A_109 = rconst(index,109)*var[108]*var[126];
        A_110 = rconst(index,110)*var[96]*var[126];
        A_111 = rconst(index,111)*var[62]*var[126];
        A_112 = rconst(index,112)*var[40]*var[126];
        A_113 = rconst(index,113)*var[102]*var[125];
        A_114 = rconst(index,114)*var[102]*var[137];
        A_115 = rconst(index,115)*var[102]*var[133];
        A_116 = rconst(index,116)*var[79]*var[126];
        A_117 = rconst(index,117)*var[110]*var[124];
        A_118 = rconst(index,118)*var[110]*var[126];
        A_119 = rconst(index,119)*var[113]*var[137];
        A_120 = rconst(index,120)*var[113]*var[133];
        A_121 = rconst(index,121)*var[113]*var[135];
        A_122 = 2e-12*var[113]*var[125];
        A_123 = 2e-12*var[113]*var[113];
        A_124 = 3e-11*var[82]*var[126];
        A_125 = rconst(index,125)*var[85]*var[126];
        A_126 = rconst(index,126)*var[99]*var[137];
        A_127 = rconst(index,127)*var[99]*var[133];
        A_128 = rconst(index,128)*var[68]*var[126];
        A_129 = 1.7e-12*var[111]*var[126];
        A_130 = 3.2e-11*var[64]*var[126];
        A_131 = rconst(index,131)*var[64];
        A_132 = rconst(index,132)*var[106]*var[124];
        A_133 = rconst(index,133)*var[106]*var[126];
        A_134 = rconst(index,134)*var[106]*var[136];
        A_135 = rconst(index,135)*var[109]*var[137];
        A_136 = rconst(index,136)*var[109]*var[133];
        A_137 = 2e-12*var[109]*var[125];
        A_138 = 2e-12*var[109]*var[109];
        A_139 = 1e-10*var[66]*var[126];
        A_140 = 1.3e-11*var[91]*var[126];
        A_141 = rconst(index,141)*var[124]*var[127];
        A_142 = rconst(index,142)*var[131]*var[134];
        A_143 = rconst(index,143)*var[134]*var[134];
        A_144 = rconst(index,144)*var[134]*var[134];
        A_145 = rconst(index,145)*var[134]*var[134];
        A_146 = rconst(index,146)*var[134]*var[134];
        A_147 = rconst(index,147)*var[39];
        A_148 = rconst(index,148)*var[97]*var[127];
        A_149 = rconst(index,149)*var[127]*var[137];
        A_150 = rconst(index,150)*var[127]*var[137];
        A_151 = rconst(index,151)*var[88]*var[127];
        A_152 = rconst(index,152)*var[126]*var[134];
        A_153 = rconst(index,153)*var[134]*var[137];
        A_154 = rconst(index,154)*var[126]*var[138];
        A_155 = rconst(index,155)*var[112]*var[126];
        A_156 = rconst(index,156)*var[133]*var[134];
        A_157 = rconst(index,157)*var[134]*var[135];
        A_158 = rconst(index,158)*var[116];
        A_159 = rconst(index,159)*var[116]*var[131];
        A_160 = rconst(index,160)*var[116]*var[127];
        A_161 = rconst(index,161)*var[98]*var[127];
        A_162 = rconst(index,162)*var[127]*var[130];
        A_163 = 5.9e-11*var[104]*var[127];
        A_164 = rconst(index,164)*var[125]*var[134];
        A_165 = 3.3e-10*var[41]*var[120];
        A_166 = 1.65e-10*var[75]*var[120];
        A_167 = rconst(index,167)*var[75]*var[126];
        A_168 = 3.25e-10*var[57]*var[120];
        A_169 = rconst(index,169)*var[57]*var[126];
        A_170 = rconst(index,170)*var[103]*var[127];
        A_171 = 8e-11*var[119]*var[127];
        A_172 = 1.4e-10*var[42]*var[120];
        A_173 = 2.3e-10*var[43]*var[120];
        A_174 = rconst(index,174)*var[124]*var[129];
        A_175 = rconst(index,175)*var[131]*var[132];
        A_176 = 2.7e-12*var[132]*var[132];
        A_177 = rconst(index,177)*var[132]*var[132];
        A_178 = rconst(index,178)*var[129]*var[137];
        A_179 = rconst(index,179)*var[132]*var[137];
        A_180 = rconst(index,180)*var[123]*var[126];
        A_181 = rconst(index,181)*var[118]*var[131];
        A_182 = rconst(index,182)*var[100]*var[126];
        A_183 = 4.9e-11*var[105]*var[129];
        A_184 = rconst(index,184)*var[132]*var[133];
        A_185 = rconst(index,185)*var[132]*var[135];
        A_186 = rconst(index,186)*var[105];
        A_187 = rconst(index,187)*var[129]*var[130];
        A_188 = rconst(index,188)*var[104]*var[129];
        A_189 = rconst(index,189)*var[125]*var[132];
        A_190 = rconst(index,190)*var[125]*var[132];
        A_191 = rconst(index,191)*var[53]*var[126];
        A_192 = rconst(index,192)*var[103]*var[129];
        A_193 = rconst(index,193)*var[119]*var[129];
        A_194 = rconst(index,194)*var[45]*var[126];
        A_195 = rconst(index,195)*var[44]*var[126];
        A_196 = 3.32e-15*var[90]*var[129];
        A_197 = 1.1e-15*var[80]*var[129];
        A_198 = rconst(index,198)*var[100]*var[127];
        A_199 = rconst(index,199)*var[132]*var[134];
        A_200 = rconst(index,200)*var[132]*var[134];
        A_201 = rconst(index,201)*var[132]*var[134];
        A_202 = 1.45e-11*var[90]*var[127];
        A_203 = rconst(index,203)*var[54]*var[126];
        A_204 = rconst(index,204)*var[55]*var[126];
        A_205 = rconst(index,205)*var[52]*var[126];
        A_206 = rconst(index,206)*var[56]*var[126];
        A_207 = rconst(index,207)*var[114]*var[126];
        A_208 = rconst(index,208)*var[114]*var[126];
        A_209 = rconst(index,209)*var[114]*var[136];
        A_210 = 1e-10*var[65]*var[126];
        A_211 = rconst(index,211)*var[81];
        A_212 = 3e-13*var[81]*var[124];
        A_213 = 5e-11*var[46]*var[137];
        A_214 = 3.3e-10*var[114]*var[127];
        A_215 = rconst(index,215)*var[114]*var[129];
        A_216 = 4.4e-13*var[114]*var[132];
        A_217 = rconst(index,217)*fix(index,0);
        A_218 = rconst(index,218)*var[124];
        A_219 = rconst(index,219)*var[124];
        A_220 = rconst(index,220)*var[128];
        A_221 = rconst(index,221)*var[88];
        A_222 = rconst(index,222)*var[60];
        A_223 = rconst(index,223)*var[135];
        A_224 = rconst(index,224)*var[133];
        A_225 = rconst(index,225)*var[136];
        A_226 = rconst(index,226)*var[136];
        A_227 = rconst(index,227)*var[83];
        A_228 = rconst(index,228)*var[76];
        A_229 = rconst(index,229)*var[101];
        A_230 = rconst(index,230)*var[73];
        A_231 = rconst(index,231)*var[104];
        A_232 = rconst(index,232)*var[130];
        A_233 = rconst(index,233)*var[130];
        A_234 = rconst(index,234)*fix(index,2);
        A_235 = rconst(index,235)*var[98];
        A_236 = rconst(index,236)*var[71];
        A_237 = rconst(index,237)*var[119];
        A_238 = rconst(index,238)*var[63];
        A_239 = rconst(index,239)*var[58];
        A_240 = rconst(index,240)*var[77];
        A_241 = rconst(index,241)*var[69];
        A_242 = rconst(index,242)*var[86];
        A_243 = rconst(index,243)*var[108];
        A_244 = rconst(index,244)*var[96];
        A_245 = rconst(index,245)*var[72];
        A_246 = rconst(index,246)*var[62];
        A_247 = rconst(index,247)*var[79];
        A_248 = rconst(index,248)*var[110];
        A_249 = rconst(index,249)*var[82];
        A_250 = rconst(index,250)*var[85];
        A_251 = rconst(index,251)*var[68];
        A_252 = rconst(index,252)*var[38];
        A_253 = rconst(index,253)*var[111];
        A_254 = rconst(index,254)*var[64];
        A_255 = rconst(index,255)*var[66];
        A_256 = rconst(index,256)*var[91];
        A_257 = rconst(index,257)*var[80];
        A_258 = rconst(index,258)*var[39];
        A_259 = rconst(index,259)*var[51];
        A_260 = rconst(index,260)*var[138];
        A_261 = rconst(index,261)*var[112];
        A_262 = rconst(index,262)*var[50];
        A_263 = rconst(index,263)*var[116];
        A_264 = rconst(index,264)*var[116];
        A_265 = rconst(index,265)*var[75];
        A_266 = rconst(index,266)*var[41];
        A_267 = rconst(index,267)*var[57];
        A_268 = rconst(index,268)*var[43];
        A_269 = rconst(index,269)*var[42];
        A_270 = rconst(index,270)*var[100];
        A_271 = rconst(index,271)*var[132];
        A_272 = rconst(index,272)*var[118];
        A_273 = rconst(index,273)*var[0];
        A_274 = rconst(index,274)*var[105];
        A_275 = rconst(index,275)*var[53];
        A_276 = rconst(index,276)*var[44];
        A_277 = rconst(index,277)*var[45];
        A_278 = rconst(index,278)*var[2];
        A_279 = rconst(index,279)*var[90];
        A_280 = rconst(index,280)*var[1];
        A_281 = rconst(index,281)*var[52];
        A_282 = rconst(index,282)*var[54];
        A_283 = rconst(index,283)*var[55];
        A_284 = rconst(index,284)*var[3];
        A_285 = rconst(index,285)*var[83]*var[128];
        A_286 = rconst(index,286)*var[83];
        A_287 = rconst(index,287)*var[112]*var[138];
        A_288 = rconst(index,288)*var[116]*var[138];
        A_289 = rconst(index,289)*var[116]*var[128];
        A_290 = rconst(index,290)*var[83]*var[138];
        A_291 = rconst(index,291)*var[118]*var[123];
        A_292 = rconst(index,292)*var[105]*var[128];
        A_293 = rconst(index,293)*var[116]*var[123];
        A_294 = rconst(index,294)*var[105]*var[138];
        A_295 = rconst(index,295)*var[112]*var[123];
        A_296 = rconst(index,296)*var[118]*var[138];
        A_297 = rconst(index,297)*var[4];
        A_298 = 2.3e-10*var[15]*var[120];
        A_299 = rconst(index,299)*var[15];
        A_300 = 1.4e-10*var[16]*var[120];
        A_301 = rconst(index,301)*var[16];
        A_302 = rconst(index,302)*var[17]*var[120];
        A_303 = rconst(index,303)*var[17]*var[120];
        A_304 = rconst(index,304)*var[17];
        A_305 = 3e-10*var[18]*var[120];
        A_306 = rconst(index,306)*var[18]*var[126];
        A_307 = rconst(index,307)*var[18];
        A_308 = rconst(index,308)*var[5];
        A_309 = rconst(index,309)*var[6];
        varDot[0] = - A_273;
        varDot[1] = - A_280;
        varDot[2] = - A_278;
        varDot[3] = - A_284;
        varDot[4] = - A_297;
        varDot[5] = - A_308;
        varDot[6] = - A_309;
        varDot[7] = A_165+ 0.9*A_166+ A_167+ 2*A_168+ 2*A_169+ A_172+ A_173+ A_191+ A_194+ A_195+ A_203+ A_204+ A_205+ A_266+ 2 *A_267+ A_268+ A_269+ A_276+ A_277+ A_278+ A_280+ A_281+ A_282+ A_283;
        varDot[8] = 2*A_172+ A_173+ A_268+ 2*A_269+ 3*A_278+ 2*A_280;
        varDot[9] = 0.09*A_166+ 2*A_203+ A_204+ A_205+ 2*A_268+ A_269;
        varDot[10] = 0.4*A_210+ A_213;
        varDot[11] = A_206;
        varDot[12] = 2*A_286;
        varDot[13] = 2*A_286;
        varDot[14] = A_299+ A_301+ A_303+ A_304+ A_307+ A_308+ A_309;
        varDot[15] = - A_298- A_299;
        varDot[16] = - A_300- A_301;
        varDot[17] = - A_302- A_303- A_304;
        varDot[18] = - A_305- A_306- A_307;
        varDot[19] = A_297;
        varDot[20] = A_11;
        varDot[21] = A_17;
        varDot[22] = 2*A_2+ 2*A_3+ A_5+ A_6+ A_7+ A_8+ A_10+ A_11+ A_17+ A_21+ A_22+ 2*A_25+ A_35+ A_41+ A_46+ A_47+ A_48+ A_52 + A_56+ A_61+ A_66+ A_70+ A_74+ A_78+ A_84+ A_94+ A_96+ A_117+ A_132+ A_134+ 2*A_142+ 2*A_143+ 2*A_144 + A_145+ A_152+ A_164+ A_166+ A_168+ 2*A_175+ 2*A_176+ 2*A_177+ A_181+ A_190+ A_199+ 2*A_200+ 2*A_201+ 2 *A_226+ 2*A_258+ A_261+ A_272+ A_285+ 3*A_286+ A_287+ A_288+ 2*A_290+ A_291+ A_293+ A_294+ A_295+ A_296;
        varDot[23] = 2*A_175+ 2*A_176+ 2*A_177+ A_181+ A_190+ 0.5*A_199+ A_200+ A_201+ A_272+ A_291+ 0.333333*A_293+ 0.333333  *A_294+ 0.5*A_295+ 0.5*A_296;
        varDot[24] = 2*A_142+ 2*A_143+ 2*A_144+ A_145+ A_152+ A_164+ A_166+ A_168+ 0.5*A_199+ A_200+ A_201+ 2*A_258+ A_261  + A_287+ 0.5*A_288+ A_290+ 0.333333*A_293+ 0.333333*A_294+ 0.5*A_295+ 0.5*A_296;
        varDot[25] = A_5+ A_6+ A_7+ A_8+ A_10+ A_11;
        varDot[26] = 2*A_25+ A_35+ A_41+ A_46+ A_47+ A_48+ A_52+ 2*A_226+ A_285+ 3*A_286+ 0.5*A_288+ A_290+ 0.333333*A_293  + 0.333333*A_294;
        varDot[27] = 2*A_2+ 2*A_3+ A_17+ A_21+ A_22+ A_56;
        varDot[28] = A_61+ A_66+ A_70+ A_74+ A_78+ A_84+ A_94+ A_96+ A_117+ A_132+ A_134;
        varDot[29] = A_8;
        varDot[30] = A_32;
        varDot[31] = A_191+ A_275+ A_278+ A_280;
        varDot[32] = 4*A_165+ A_166+ A_167+ 3*A_168+ 3*A_169+ 2*A_172+ 3*A_173+ A_265+ 4*A_266+ 3*A_267+ 3*A_268+ 2*A_269  + A_280;
        varDot[33] = A_60;
        varDot[34] = A_14+ A_19+ A_24+ A_32+ A_36+ A_37+ A_60+ A_73+ A_81+ A_82+ A_98+ A_102+ A_106+ A_115+ A_120+ A_127+ A_136 + A_150+ A_182+ A_207+ A_208+ 0.4*A_210+ A_214+ A_215+ 2*A_217+ A_222+ A_224+ 0.333*A_230+ A_234+ A_259 + A_262+ A_273;
        varDot[35] = A_73+ A_82+ A_98+ A_102+ A_106+ A_115+ A_120+ A_127+ A_136;
        varDot[36] = 3*A_194+ 2*A_195+ A_203+ 2*A_204+ A_205+ 2*A_276+ 3*A_277+ A_281+ A_282+ 2*A_283;
        varDot[37] = A_281+ 2*A_282+ A_283;
        varDot[38] = 0.8*A_128- A_252;
        varDot[39] = A_146- A_147- A_258;
        varDot[40] = - A_112;
        varDot[41] = - A_165- A_266;
        varDot[42] = - A_172- A_269;
        varDot[43] = - A_173- A_268;
        varDot[44] = - A_195- A_276;
        varDot[45] = - A_194- A_277;
        varDot[46] = A_212- A_213;
        varDot[47] = - A_40;
        varDot[48] = - A_69;
        varDot[49] = - A_93;
        varDot[50] = - A_262+ A_290;
        varDot[51] = A_145+ A_199- A_259;
        varDot[52] = - A_205- A_281;
        varDot[53] = - A_191- A_275;
        varDot[54] = - A_203- A_282;
        varDot[55] = - A_204- A_283;
        varDot[56] = - A_206+ 0.6*A_210+ A_211;
        varDot[57] = - A_168- A_169- A_267;
        varDot[58] = - A_90+ A_140- A_239;
        varDot[59] = - A_19- A_24- A_27+ A_224;
        varDot[60] = - A_21- A_22+ A_27+ A_46- A_222;
        varDot[61] = A_51- A_54;
        varDot[62] = 0.04*A_98- A_111- A_246;
        varDot[63] = A_80- A_89- A_238;
        varDot[64] = A_121- A_130- A_131- A_254;
        varDot[65] = A_208- A_210+ A_216;
        varDot[66] = A_135- A_139- A_255;
        varDot[67] = A_101- A_103;
        varDot[68] = A_126- A_128- A_251;
        varDot[69] = A_97- A_100- A_241;
        varDot[70] = A_49- A_51- A_53+ A_54;
        varDot[71] = A_72- A_76- A_236;
        varDot[72] = A_105- A_108- A_245;
        varDot[73] = A_34- A_38- A_39- A_230;
        varDot[74] = - A_79+ A_81+ A_86+ 0.18*A_87;
        varDot[75] = - 0.9*A_166- A_167- A_265;
        varDot[76] = A_31- A_36+ A_52- A_228;
        varDot[77] = A_83- A_91- A_92- A_240;
        varDot[78] = - A_68+ 0.23125*A_70+ 0.22*A_94+ 0.45*A_117+ 0.28*A_132;
        varDot[79] = A_114- A_116- A_247;
        varDot[80] = A_143+ A_160- A_197+ A_202- A_257+ A_287+ A_288;
        varDot[81] = A_207+ A_209- A_211- A_212+ A_214+ A_215;
        varDot[82] = A_119- A_124- A_249;
        varDot[83] = A_29- A_30- A_227- A_285- A_286- A_290;
        varDot[84] = A_41+ A_42+ A_47- A_48- A_49;
        varDot[85] = 0.88*A_113+ 0.56*A_115+ 0.85*A_116- A_125+ A_129+ 0.67*A_247- A_250+ 0.67*A_253;
        varDot[86] = 0.96*A_98+ A_99+ 0.7*A_100- A_104+ A_111+ A_241- A_242+ A_246;
        varDot[87] = A_43- A_50- A_51- A_52+ A_53- A_55;
        varDot[88] = A_16- A_18+ 0.13875*A_70+ 0.09*A_132- A_151- A_221;
        varDot[89] = - A_58+ A_63+ 0.25*A_75+ 0.03*A_94+ 0.2*A_99+ 0.5*A_107+ 0.18*A_113+ 0.25*A_122+ 0.25*A_137;
        varDot[90] = - A_196+ A_197+ A_198+ A_201- A_202- A_279+ A_293+ A_294+ A_295+ A_296;
        varDot[91] = A_134+ 0.044*A_136- A_140- A_256;
        varDot[92] = A_40- A_41- A_42- A_43- A_44- A_45- A_46- A_47+ A_48;
        varDot[93] = 0.82*A_93- A_97- A_98- A_99+ 0.3*A_100;
        varDot[94] = A_104- A_105- A_106- A_107+ 0.3*A_108;
        varDot[95] = A_65+ A_66- A_67+ 0.63*A_70+ A_90+ A_91+ 0.31*A_94+ A_110+ 0.22*A_117+ 0.25*A_120+ 0.125*A_122+ 0.5*A_123 + 0.14*A_132+ A_162+ A_187+ A_232+ A_233+ A_234+ A_235+ A_237+ A_239+ A_244+ A_248+ 0.25*A_249;
        varDot[96] = 0.04*A_94+ 0.5*A_107+ 0.7*A_108+ A_109- A_110+ 0.9*A_117+ 0.5*A_120+ 0.5*A_122+ A_123+ 0.25*A_137- A_244  + 0.5*A_249;
        varDot[97] = - A_6- A_9+ A_13+ 0.05*A_56- A_148+ A_232+ 0.69*A_235;
        varDot[98] = - A_56- A_57+ 0.06*A_94- A_161- A_235;
        varDot[99] = A_125- A_126- A_127+ 0.2*A_128;
        varDot[100] = A_177- A_182+ A_183+ A_196- A_198- A_270+ A_291;
        varDot[101] = A_33- A_37+ A_66+ A_78+ A_209- A_229+ 2*A_285+ A_288+ A_289+ A_290+ A_292+ A_293+ A_294;
        varDot[102] = A_112- A_113- A_114- A_115+ 0.15*A_116;
        varDot[103] = - A_70- A_71- A_170- A_192;
        varDot[104] = A_59- A_64- A_163- A_188- A_231;
        varDot[105] = - A_183+ A_185- A_186- A_274- A_292- A_294;
        varDot[106] = - A_132- A_133- A_134;
        varDot[107] = - A_94- A_95- A_96;
        varDot[108] = 0.5*A_103+ 0.2*A_107- A_109+ 0.25*A_120+ 0.375*A_122+ A_123+ A_130+ 0.25*A_137+ A_140- A_243+ 0.25*A_249 + A_254;
        varDot[109] = A_133- A_135- A_136- A_137- 2*A_138;
        varDot[110] = - A_117- A_118+ 0.65*A_132+ 0.956*A_136+ 0.5*A_137+ 2*A_138+ A_139- A_248+ A_255+ A_256;
        varDot[111] = A_96+ 0.02*A_102+ 0.16*A_115+ 0.015*A_127- A_129- A_253;
        varDot[112] = A_153- A_155- A_261- A_287+ A_289- A_295;
        varDot[113] = A_118- A_119- A_120- A_121- A_122- 2*A_123+ A_124+ A_131+ 0.1*A_132;
        varDot[114] = - A_207- A_208- A_209- A_214- A_215- A_216;
        varDot[115] = 0.666667*A_71+ A_95- A_101- A_102+ 0.5*A_103+ 0.666667*A_170+ 0.666667*A_192;
        varDot[116] = A_157- A_158- A_159- A_160- A_263- A_264- A_288- A_289- A_293;
        varDot[117] = A_69- A_72- A_73- A_74- A_75+ 0.3*A_76- A_87+ 0.18*A_93+ 0.06*A_94+ 0.12*A_113+ 0.28*A_115+ 0.33*A_247  + A_250+ 0.33*A_253;
        varDot[118] = A_179- A_181+ A_182+ A_189- A_272- A_291+ A_292- A_296;
        varDot[119] = A_73+ A_74+ 0.75*A_75+ 0.7*A_76- A_77- A_78+ A_87+ 0.47*A_94+ 0.98*A_102+ 0.12*A_113+ 0.28*A_115+ 0.985  *A_127- A_171- A_193+ A_236- A_237+ 0.33*A_247+ A_251+ 0.33*A_253;
        varDot[120] = - A_0- A_2- A_6- A_17- A_20- A_21- A_22- A_56- A_165- A_166- A_168- A_172- A_173+ A_218+ A_222;
        varDot[121] = A_77+ A_78- A_80- A_81- A_82- A_83- A_84- A_85- A_86- A_87- 2*A_88+ A_89+ A_92+ 0.23*A_94+ A_106+ 0.3 *A_107+ A_110+ 0.1*A_117+ 0.25*A_120+ 0.125*A_122+ 0.985*A_127+ 0.1*A_132+ A_171+ A_193+ A_240+ A_242 + A_243+ A_244+ A_245+ A_248+ 0.25*A_249+ A_250+ A_251+ 2*A_252;
        varDot[122] = - A_4- A_5+ A_6+ A_7+ A_9- A_12- A_13- A_14+ 0.4*A_56+ A_67+ A_148+ 0.09*A_166+ A_220+ A_233+ 0.31*A_235 + A_260;
        varDot[123] = A_178- A_180+ A_187+ A_188+ A_192+ A_193+ A_215- A_291- A_293- A_295;
        varDot[124] = A_1- A_2- A_3- A_5- A_8- A_11- A_23- A_26- A_41- A_48- A_70+ A_81- A_94- A_117- A_132- A_141- A_174 - A_212- A_218- A_219;
        varDot[125] = 0.75*A_56+ A_57- A_59- A_60- A_61- 2*A_62- 2*A_63+ 0.7*A_64- A_75+ A_79+ A_82+ A_84- A_86+ 0.82*A_87+ 2 *A_88+ 0.07*A_94- A_99- A_107- A_113- A_122+ 0.08*A_132- A_137+ A_161- A_164+ A_188- A_189- A_190+ 0.6 *A_210+ A_211+ A_237+ A_238+ A_242+ A_265+ A_275+ A_284;
        varDot[126] = A_5+ A_6- A_7- A_8- A_9+ A_10+ A_11+ 2*A_12- A_15+ 2*A_17- A_18- A_31+ A_32- A_33+ A_35- A_36- A_37 - A_39- A_40+ A_42+ A_44- A_50- A_53- A_54+ 0.75*A_56- A_57- A_58- 0.7*A_64- A_65- A_67- A_68- A_69+ 0.13 *A_70- A_71- 0.3*A_76- A_77- A_79- A_89- A_90- A_91- A_93+ 0.33*A_94- A_95- 0.3*A_100- 0.5*A_103- A_104 - 0.3*A_108- A_109- A_110- A_111- A_112- 0.15*A_116+ 0.19*A_117- A_118- A_124- A_125- 0.2*A_128- A_129 - A_130+ 0.25*A_132- A_133- A_140+ A_150- A_152- A_154- A_155+ A_163- A_167+ A_168- A_169- A_180+ A_181 - A_182- A_191- A_194- A_195- A_203- A_204- A_205- A_206- A_207- A_208- A_210+ A_220+ 2*A_221+ A_228+ A_229 + 0.333*A_230+ A_231+ A_236+ A_238+ A_241+ A_245+ A_247+ A_249+ A_251+ A_255+ A_261+ A_272;
        varDot[127] = - A_141+ A_142+ 2*A_144+ A_145- A_148- A_149- A_150- A_151+ 0.94*A_152+ A_154+ A_156- A_160- A_161- A_162 - A_163+ A_164+ 3*A_165+ 0.35*A_166+ A_167+ 3*A_168+ 3*A_169- A_170- A_171+ A_172+ 2*A_173+ A_196+ A_197 - A_198+ A_200- A_202- A_214+ 2*A_257+ 2*A_258+ A_260+ A_261+ A_262+ A_263+ A_265+ 4*A_266+ 3*A_267+ A_268 + A_269+ A_279+ A_280+ A_281+ 2*A_282+ A_283;
        varDot[128] = A_9+ A_14+ A_15- A_17+ A_18+ A_36+ A_37+ A_39+ A_40+ A_43+ A_45+ A_46+ A_50+ A_53+ A_54+ A_57+ A_64 + A_65+ A_68+ A_69+ A_77+ A_79+ A_89+ A_91+ A_93+ A_103+ A_104+ A_112+ 0.85*A_116+ A_129+ A_154+ A_155 + A_167+ A_169+ A_180+ A_191+ A_194+ A_195+ A_203+ A_204+ A_205- A_220+ 1.155*A_235- A_285+ A_287- A_289 + A_291- A_292+ A_295+ A_296;
        varDot[129] = - A_174+ A_175+ 2*A_176- A_178+ A_180+ A_182- A_183+ A_184- A_187- A_188+ A_190+ A_191- A_192- A_193+ 3 *A_194+ 2*A_195- A_196- A_197+ A_198+ A_199+ A_200+ A_202+ A_203+ 2*A_204+ A_205- A_215+ A_216+ 2*A_270 + A_271+ A_272+ A_273+ 0.85*A_274+ A_275+ 2*A_276+ 3*A_277+ A_278+ A_279+ A_280+ A_281+ A_282+ 2*A_283;
        varDot[130] = 0.25*A_56+ A_58+ A_60+ A_61+ 2*A_62+ A_63+ 0.3*A_64- A_65- A_66+ 1.13875*A_70+ 0.75*A_75+ A_85+ A_86  + A_90+ A_91+ 0.57*A_94+ 0.8*A_99+ 0.98*A_102+ A_106+ 0.8*A_107+ 0.68*A_113+ 0.75*A_120+ 1.125*A_122+ 0.5 *A_123+ 0.58*A_132+ 0.956*A_136+ 1.25*A_137+ A_138- A_162+ A_163+ A_164- A_187+ A_189+ A_190+ A_207+ A_209 + A_210+ A_214+ A_215+ A_231- A_232- A_233+ A_239+ A_243+ A_245+ A_248+ 0.75*A_249+ A_255+ A_256;
        varDot[131] = A_0- A_1- A_3- A_7- A_10+ A_14+ A_19+ A_20+ A_24- A_25+ A_27- A_142- A_159+ 0.1*A_166- A_175- A_181+ 2 *A_217+ A_219+ A_223+ A_224+ A_225+ A_234+ A_259+ A_271;
        varDot[132] = A_174- A_175- 2*A_176- 2*A_177- A_179+ A_181- A_184- A_185+ A_186- A_189- A_190- A_199- A_200- A_201  - A_216- A_271+ 0.15*A_274;
        varDot[133] = A_19+ 2*A_21- A_23- A_24+ A_25- A_28- A_31- A_32- A_44- A_45+ A_47+ A_50+ A_51+ A_52+ A_55- A_60- A_73 - A_82- A_98- A_102- A_106- A_115- A_120- A_127- A_136- A_156- A_184+ A_223- A_224+ A_226+ A_228;
        varDot[134] = A_141- A_142- 2*A_143- 2*A_144- 2*A_145- 2*A_146+ 2*A_147+ A_150- A_152- A_153+ A_155- A_156- A_157  + A_158+ A_159- A_164+ A_165+ 0.46*A_166+ A_172+ A_173- A_199- A_200- A_201+ A_259+ A_264;
        varDot[135] = A_23- A_25- A_26- A_27+ 2*A_28- A_29+ A_30+ A_32- A_33- A_34+ A_35+ A_36+ A_38+ A_39- A_46- A_47- A_52 + A_60+ A_61+ A_73+ A_74+ A_82- A_83+ A_84+ A_90+ A_91+ A_92+ 0.96*A_98+ 0.98*A_102+ A_106+ A_111+ 0.84 *A_115+ A_120- A_121+ 0.985*A_127+ A_129+ A_130+ A_131+ 0.956*A_136+ A_156- A_157+ A_158+ A_184- A_185 + A_186- A_223+ A_225+ A_227+ A_229+ 0.667*A_230+ A_239+ A_240+ A_246+ A_253+ A_254+ A_256+ A_262+ A_264 + A_273+ 0.15*A_274;
        varDot[136] = A_26- A_28- A_29+ A_30- A_35+ A_37- A_61- A_66- A_74- A_78- A_84- A_96- A_134+ A_159+ A_160+ A_183- A_209 - A_225- A_226+ A_227+ 0.333*A_230+ A_263+ 0.85*A_274;
        varDot[137] = A_4+ A_8- A_10- A_11- A_12- A_13- A_14- A_15- 2*A_16+ A_18- A_32- A_34- A_35+ A_38- A_42- A_43+ A_44 + A_55+ A_58- A_59+ A_60+ A_61+ 2*A_62+ A_65+ A_66+ A_68+ 0.13*A_70- A_72+ A_73+ A_74+ A_75- A_80- A_81 + A_85+ 0.82*A_87+ 0.26*A_94- A_97+ 0.96*A_98+ 0.8*A_99- A_101+ 0.98*A_102- A_105+ 0.3*A_107+ A_109+ 1.23 *A_113- A_114+ 0.56*A_115+ 0.32*A_117- A_119+ 0.75*A_120+ 0.875*A_122+ A_123- A_126+ 0.25*A_132- A_135 + 0.956*A_136+ A_137+ A_138- A_149- A_150+ A_151+ 0.94*A_152- A_153+ A_162+ A_164- A_178- A_179+ A_187 + A_190+ A_206+ A_208+ 0.4*A_210- A_213+ 0.667*A_230+ A_231+ A_233+ A_236+ A_237+ A_241+ A_243+ A_244 + A_246+ 0.67*A_247+ A_248+ 0.75*A_249+ 0.67*A_253+ A_255+ A_256;
        varDot[138] = A_148+ A_149+ A_151+ 0.06*A_152- A_154+ A_161+ A_162+ A_163+ A_170+ A_171+ A_214- A_260- A_287- A_288 - A_290- A_294- A_296;
    }
}

__device__ void ros_FunTimeDerivative(const double T, double roundoff, double * __restrict__ var, const double * __restrict__ fix, 
                                      const double * __restrict__ rconst, double *dFdT, double *Fcn0, int &Nfun, 
                                      const double * __restrict__ khet_st, const double * __restrict__ khet_tr,
                                      const double * __restrict__ jx,
                                      const int VL_GLO)
{
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    const double DELTAMIN = 1.0E-6;
    double delta,one_over_delta;

    delta = sqrt(roundoff)*fmax(DELTAMIN,fabs(T));
    one_over_delta = 1.0/delta;

    Fun(var, fix, rconst, dFdT, Nfun, VL_GLO);

    for (int i=0; i < NVAR; i++){
        dFdT(index,i) = (dFdT(index,i) - Fcn0(index,i)) * one_over_delta;
    }
}

__device__  static  int ros_Integrator(double * __restrict__ var, const double * __restrict__ fix, const double Tstart, const double Tend, double &T,
        //  Rosenbrock method coefficients
        const int ros_S, const double * __restrict__ ros_M, const double * __restrict__ ros_E, const double * __restrict__ ros_A, const double * __restrict__  ros_C, 
        const double * __restrict__ ros_Alpha, const double * __restrict__ ros_Gamma, const double ros_ELO, const int * ros_NewF, 
        //  Integration parameters
        const int autonomous, const int vectorTol, const int Max_no_steps, 
        const double roundoff, const double Hmin, const double Hmax, const double Hstart, double &Hexit, 
        const double FacMin, const double FacMax, const double FacRej, const double FacSafe, 
        //  Status parameters
        int &Nfun, int &Njac, int &Nstp, int &Nacc, int &Nrej, int &Ndec, int &Nsol, int &Nsng,
        //  cuda global mem buffers              
        const double * __restrict__ rconst,  const double * __restrict__ absTol, const double * __restrict__ relTol, double * __restrict__ varNew, double * __restrict__ Fcn0, 
        double * __restrict__ K, double * __restrict__ dFdT, double * __restrict__ jac0, double * __restrict__ Ghimj, double * __restrict__ varErr,
        // for update_rconst
        const double * __restrict__ khet_st, const double * __restrict__ khet_tr,
        const double * __restrict__ jx,
        // VL_GLO
        const int VL_GLO)
{
    int index = blockIdx.x*blockDim.x+threadIdx.x;

    double H, Hnew, HC, HG, Fac; // Tau - not used
    double Err; //*varErr;
    int direction;
    int rejectLastH, rejectMoreH;
    const double DELTAMIN = 1.0E-5;

    //   ~~~>  Initial preparations
    T = Tstart;
    Hexit = 0.0;
    H = fmin(Hstart,Hmax);
    if (fabs(H) <= 10.0*roundoff) 
        H = DELTAMIN;

    if (Tend  >=  Tstart)
    {
        direction = + 1;
    }
    else
    {
        direction = - 1;
    }

    rejectLastH=0;
    rejectMoreH=0;



    //   ~~~> Time loop begins below

    // TimeLoop: 
    while((direction > 0) && ((T- Tend)+ roundoff <= ZERO) || (direction < 0) && ((Tend-T)+ roundoff <= ZERO))
    {
        if (Nstp > Max_no_steps) //  Too many steps
            return -6;
        //  Step size too small
        if (H <= roundoff){  //  Step size too small
            //if (((T+ 0.1*H) == T) || (H <= roundoff)) {
            return -7;
        }

        //   ~~~>  Limit H if necessary to avoid going beyond Tend
        Hexit = H;
        H = fmin(H,fabs(Tend-T));

        //   ~~~>   Compute the function at current time
        Fun(var, fix, rconst, Fcn0, Nfun, VL_GLO);	/// VAR READ - Fcn0 Write

        //   ~~~>  Compute the function derivative with respect to T
        if (!autonomous)
            ros_FunTimeDerivative(T, roundoff, var, fix, rconst, dFdT, Fcn0, Nfun, khet_st, khet_tr, jx,  VL_GLO); /// VAR READ - fcn0 read

        //   ~~~>   Compute the Jacobian at current time
        Jac_sp(var, fix, rconst, jac0, Njac, VL_GLO);   /// VAR READ 

        //   ~~~>  Repeat step calculation until current step accepted
        // UntilAccepted: 
        while(1)
        {
            ros_PrepareMatrix(H, direction, ros_Gamma[0], jac0, Ghimj, Nsng, Ndec, VL_GLO);
            //   ~~~>   Compute the stages
            // Stage: 
            for (int istage=0; istage < ros_S; istage++)
            {
                //   For the 1st istage the function has been computed previously
                if (istage == 0)
                {
                    for (int i=0; i<NVAR; i++){
                        varNew(index,i) = Fcn0(index,i);				// FCN0 Read
                    }
                }
                else if(ros_NewF[istage])
                {
                        for (int i=0; i<NVAR; i++){		
                            varNew(index,i) = var(index,i);
                        }

                    for (int j=0; j < (istage); j++){
                        for (int i=0; i<NVAR; i++){		
                            varNew(index,i) = K(index,j,i)*ros_A[(istage)*(istage-1)/2 + j]  + varNew(index,i);
                        }
                    }
                    Fun(varNew, fix, rconst, varNew, Nfun,VL_GLO); // FCN <- varNew / not overlap 
		} 

		for (int i=0; i<NVAR; i++)		
			K(index,istage,i)  = varNew(index,i);

		for (int j=0; j<(istage); j++)
		{
			HC = ros_C[(istage)*(istage-1)/2 + j]/(direction*H);
			for (int i=0; i<NVAR; i++){
				double tmp = K(index,j,i);
				K(index,istage,i) += tmp*HC;
			}
		}

                if ((!autonomous) && (ros_Gamma[istage] ))
                {
                    HG = direction*H*ros_Gamma[istage];
                    for (int i=0; i<NVAR; i++){
                        K(index,istage,i) += dFdT(index,i)*HG;
		     }
                }
		//	   R   ,RW, RW,  R,        R 
                ros_Solve(Ghimj, K, Nsol, istage, ros_S);


            } // Stage

            //  ~~~>  Compute the new solution
	    for (int i=0; i<NVAR; i++){
		    double tmpNew  = var(index,i); 					/// VAR READ
		    double tmpErr  = ZERO;

		    for (int j=0; j<ros_S; j++){
		    	    double tmp = K(index,j,i);

#ifdef DEBUG
			    if (isnan(tmp)){
			    	printf("Solver detected NAN!");
			    	tmp = 0;
			    }
#endif
			    tmpNew += tmp*ros_M[j];
			    tmpErr += tmp*ros_E[j];
		    }
		    varNew(index,i) = tmpNew;			// varNew is killed
		    varErr(index,i) = tmpErr;
	    }

            Err = ros_ErrorNorm(var, varNew, varErr, absTol, relTol, vectorTol);   /// VAR-varNew READ


//  ~~~> New step size is bounded by FacMin <= Hnew/H <= FacMax
            Fac  = fmin(FacMax,fmax(FacMin,FacSafe/pow(Err,ONE/ros_ELO)));
            Hnew = H*Fac;

//  ~~~>  Check the error magnitude and adjust step size
            Nstp = Nstp+ 1;
            if((Err <= ONE) || (H <= Hmin)) // ~~~> Accept step
            {
                Nacc = Nacc + 1;
                for (int j=0; j<NVAR ; j++)
                    var(index,j) =  fmax(varNew(index,j),ZERO);  /////////// VAR WRITE - last VarNew read

                T = T +  direction*H;
                Hnew = fmax(Hmin,fmin(Hnew,Hmax));
                if (rejectLastH)   // No step size increase after a rejected step
                    Hnew = fmin(Hnew,H);
                rejectLastH = 0;
                rejectMoreH = 0;
                H = Hnew;

            	break;  //  EXIT THE LOOP: WHILE STEP NOT ACCEPTED
            }
            else      // ~~~> Reject step
            {
                if (rejectMoreH)
                    Hnew = H*FacRej;
                rejectMoreH = rejectLastH;
                rejectLastH = 1;
                H = Hnew;
                if (Nacc >= 1)
                    Nrej += 1;
            } //  Err <= 1
        } // UntilAccepted
    } // TimeLoop
//  ~~~> Succesful exit
    return 0; //  ~~~> The integration was successful
}

typedef struct {
 double ros_A[15];
 double ros_C[15];
 int   ros_NewF[8];
 double ros_M[6];
 double ros_E[6];
 double ros_Alpha[6];
 double ros_Gamma[6];
 double ros_ELO;
 int    ros_S;
} ros_t;

/*
 * Lookup tables for different ROS for branch elimination. It is much faster in GPU.
 */
__device__ __constant__  ros_t ros[5] = {
    {       
        {.58578643762690495119831127579030,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, /* ros_A */
        {-1.17157287525380990239662255158060,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, /* ros_C */
        {1,1,0,0,0,0,0,0}, /* ros_NewF */
        {.87867965644035742679746691368545,.29289321881345247559915563789515,0,0,0,0}, /* ros_M */
        {.29289321881345247559915563789515,.29289321881345247559915563789515,0,0,0,0}, /* ros_E */
        {0,1.0,0,0,0,0}, /* ros_Alpha */
        {1.70710678118654752440084436210485,-1.70710678118654752440084436210485,0,0,0,0},  /* ros_Gamma */
        2.0, /* ros_ELO */
        2, /* ros_S*/
    }, /* Ros2 */
    {       
        {1.0,1.0,0,0,0,0,0,0,0,0,0,0,0,0,0}, /* ros_A */
        {-0.10156171083877702091975600115545E+01, 0.40759956452537699824805835358067E+01,0.92076794298330791242156818474003E+01,0,0,0,0,0,0,0,0,0,0,0,0}, /* ros_C */
        {1,1,0,0,0,0,0,0}, /* ros_NewF */
        {0.1E+01,0.61697947043828245592553615689730E+01,-0.42772256543218573326238373806514E+00,0,0,0}, /* ros_M */
        {0.5E+00,- 0.29079558716805469821718236208017E+01,0.22354069897811569627360909276199E+00,0,0,0}, /* ros_E */
        {0.0E+00,0.43586652150845899941601945119356E+00,0.43586652150845899941601945119356E+00,0,0,0}, /* ros_Alpha */
        {0.43586652150845899941601945119356E+00,0.24291996454816804366592249683314E+00,0.21851380027664058511513169485832E+01,0,0,0},  /* ros_Gamma */
        3.0, /* ros_ELO */
        3
    }, /* Ros3 */
    {       
        {0.2000000000000000E+01, 0.1867943637803922E+01, 0.2344449711399156E+00, 0.1867943637803922E+01, 0.2344449711399156E+00,0,0,0,0,0,0,0,0,0,0}, /* ros_A */
        {-0.7137615036412310E+01,0.2580708087951457E+01,0.6515950076447975E+00, - 0.2137148994382534E+01, - 0.3214669691237626E+00, - 0.6949742501781779E+00 ,0,0,0,0,0,0,0,0,0}, /* ros_C */
        {1,1,1,0,0,0,0,0}, /* ros_NewF */
        {0.2255570073418735E+01, 0.2870493262186792E+00, 0.4353179431840180E+00, 0.1093502252409163E+01,0,0}, /* ros_M */
        { -0.2815431932141155E+00, -0.7276199124938920E-01, -0.1082196201495311E+00, -0.1093502252409163E+01, 0, 0}, /* ros_E */
        {0.0, 0.1145640000000000E+01, 0.6552168638155900E+00, 0.6552168638155900E+00,0,0}, /* ros_Alpha */
        { 0.5728200000000000E+00, -0.1769193891319233E+01, 0.7592633437920482E+00, -0.1049021087100450E+00,0,0},  /* ros_Gamma */
        4.0, /* ros_ELO */
        4
    }, /* Ros4 */
    {       
        { 0.0E+00, 2.0E+00, 0.0E+00, 2.0E+00, 0.0E+00, 1.0E+00, 0,0,0,0,0,0,0,0,0}, /* ros_A */
        { 4.0E+00, 1.0E+00, - 1.0E+00,  1.0E+00, - 1.0E+00, - 2.66666666666666666666666666666666, 0,0,0,0,0,0,0,0,0}, /* ros_C */
        {1,0,1,1,0,0,0,0}, /* ros_NewF */
        {2.0,0,1.0,1.0,0,0}, /* ros_M */
        {0,0,0,1.0,0,0}, /* ros_E */
        {0,0,1.0,1.0,0,0}, /* ros_Alpha */
        {0.5,1.5,0,0,0,0},  /* ros_Gamma */
        3.0, /* ros_ELO */
        4
    }, /* Rodas3 */

    { 
        {
            0.1544000000000000E+01,  0.9466785280815826E+00, 0.2557011698983284E+00, 0.3314825187068521E+01,
            0.2896124015972201E+01,  0.9986419139977817E+00, 0.1221224509226641E+01, 0.6019134481288629E+01,
            0.1253708332932087E+02, -0.6878860361058950E+00, 0.1221224509226641E+01, 0.6019134481288629E+01,
            0.1253708332932087E+02, -0.6878860361058950E+00, 1.0E+00},  /* ros_A */ 

        {
            -0.5668800000000000E+01, -0.2430093356833875E+01, -0.2063599157091915E+00, -0.1073529058151375E+00,  
            -0.9594562251023355E+01, -0.2047028614809616E+02,  0.7496443313967647E+01, -0.1024680431464352E+02,  
            -0.3399990352819905E+02,  0.1170890893206160E+02,  0.8083246795921522E+01, -0.7981132988064893E+01,  
            -0.3152159432874371E+02,  0.1631930543123136E+02, -0.6058818238834054E+01}, /* ros_C */
        {1,1,1,1,1,1,0,0}, /* ros_NewF */
        {0.1221224509226641E+01,0.6019134481288629E+01,0.1253708332932087E+02,- 0.6878860361058950E+00,1,1}, /* ros_M */
        {0,0,0,0,0,1.0}, /* ros_E */
        {0.000,  0.386,  0.210,  0.630,  1.000, 1.000}, /* ros_Alpha */
        {0.2500000000000000E+00,  -0.1043000000000000E+00,  0.1035000000000000E+00,  0.3620000000000023E-01, 0, 0},  /* ros_Gamma */
        4.0, /* ros_ELO */
        6
    } /* Rodas4 */



};



//__device__ double rconst_local[MAX_VL_GLO*NREACT];

/* Initialize rconst local  */
//__device__ double * rconst_local;


__device__ double k_3rd(double temp, double cair, double k0_300K, double n, double kinf_300K, double m, double fc)
    /*
 *    
 * temp        temperature [K]
 * cair        air concentration [molecules/cm3]
 * k0_300K     low pressure limit at 300 K
 * n           exponent for low pressure limit
 * kinf_300K   high pressure limit at 300 K
 * m           exponent for high pressure limit
 * fc          broadening factor (usually fc=0.6)
 * 
 */
{

    double zt_help, k0_T, kinf_T, k_ratio, k_3rd_r;

    zt_help = 300.0/temp;
    k0_T    = k0_300K   *pow(zt_help,n) *cair;
    kinf_T  = kinf_300K *pow(zt_help,m);
    k_ratio = k0_T/kinf_T;
    k_3rd_r   = k0_T/(1.0+ k_ratio)*pow(fc,1.0/(1.0+ pow(log10(k_ratio),2)));
    return k_3rd_r;
}

__device__ double k_3rd_iupac(double temp, double cair, double k0_300K, double n, double kinf_300K, double m, double fc)
/*
 *    
 * temp        temperature [K]
 * cair        air concentration [molecules/cm3]
 * k0_300K     low pressure limit at 300 K
 * n           exponent for low pressure limit
 * kinf_300K   high pressure limit at 300 K
 * m           exponent for high pressure limit
 * fc          broadening factor (e.g. 0.45 or 0.6...)
 * nu          N
 * 
 */
{   
 
    double zt_help, k0_T, kinf_T, k_ratio, nu, k_3rd_iupac_r;
    zt_help = 300.0/temp;
    k0_T    = k0_300K   *pow(zt_help,n) *cair;
    kinf_T  = kinf_300K *pow(zt_help,m);
    k_ratio = k0_T/kinf_T;
    nu      = 0.75- 1.27*log10(fc);
    k_3rd_iupac_r = k0_T/(1.0+ k_ratio)*pow(fc,1.0/(1.0+ pow(log10(k_ratio)/nu,2)));
    return k_3rd_iupac_r;
}




double * temp_gpu;
double * press_gpu;
double * cair_gpu;


__device__ void  update_rconst(const double * __restrict__ var, 
 			       const double * __restrict__ khet_st, const double * __restrict__ khet_tr,
 			       const double * __restrict__ jx, double * __restrict__ rconst, 
			       const double * __restrict__ temp_gpu, 
			       const double * __restrict__ press_gpu, 
			       const double * __restrict__ cair_gpu, 
			       const int VL_GLO)
{
    int index = blockIdx.x*blockDim.x+threadIdx.x;

    /* Set local buffer */

    {
        const double temp_loc  = temp_gpu[index];
        const double press_loc = press_gpu[index];
        const double cair_loc  = cair_gpu[index];

        double k_HO2_HO2, k_NO3_NO2, k_NO2_HO2, k_HNO3_OH, k_CH3OOH_OH, k_ClO_ClO, k_BrO_NO2, k_I_NO2, k_DMS_OH, k_CH2OO_SO2, k_O3s, beta_null_CH3NO3, beta_inf_CH3NO3, beta_CH3NO3, k_NO2_CH3O2, k_C6H5O2_NO2, k_CH2OO_NO2, beta_C2H5NO3, alpha_NO_HO2, beta_NO_HO2, k0_NO_HO2, k2d_NO_HO2, k1d_NO_HO2, k2w_NO_HO2, k1w_NO_HO2, k_PrO2_HO2, k_PrO2_NO, k_PrO2_CH3O2, G7402a_yield, k_CH3CO3_NO2, k_PAN_M, KRO2NO, KRO2HO2[12], KAPNO, KRO2NO3, KNO3AL, KAPHO2, k_CH3O2, k_RO2RCO3, k_RO2pRO2, k_RO2sRO2, k_RO2tRO2, k_RO2pORO2, k_RO2sORO2, k_RO2tORO2, k_RO2LISOPACO2, k_RO2ISOPBO2, k_RO2ISOPDO2, k_p, k_s, k_t, k_rohro, k_co2h, k_adp, k_ads, k_adt, KHSB, KHSD, K16HSZ14, K16HSZ41, K16HS, K15HSDHB, K14HSAL, K15HS24VYNAL, K15HS42VYNAL, KHYDEC, k_CH2CHOH_OH_HCOOH, k_CH2CHOH_OH_ALD, k_CH2CHOH_HCOOH, k_ALD_HCOOH, J_IC3H7NO3, J_ACETOL, J_HPALD, J_KETENE, RO2, k1_RO2RCO3, k1_RO2pRO2, k1_RO2sRO2, k1_RO2tRO2, k1_RO2pORO2, k1_RO2sORO2, k1_RO2tORO2, k1_RO2LISOPACO2, k1_RO2ISOPBO2, k1_RO2ISOPDO2;

        k_HO2_HO2 = (3.0E-13 *exp(460. / temp_loc)+ 2.1E-33 *exp(920. / temp_loc) *cair_loc) * (1.+ 1.4E-21 *exp(2200. / temp_loc) *var[ind_H2O]);
        k_NO3_NO2 = k_3rd(temp_loc , cair_loc , 2.4E-30 , 3.0 , 1.6E-12 , - 0.1 , 0.6);
        k_NO2_HO2 = k_3rd(temp_loc , cair_loc , 1.9E-31 , 3.4 , 4.0E-12 , 0.3 , 0.6);
        k_HNO3_OH = 1.32E-14 *exp(527. / temp_loc) + 1. / (1. / (7.39E-32 *exp(453. / temp_loc) *cair_loc) + 1. / (9.73E-17 *exp(1910. / temp_loc)));
        k_CH3OOH_OH = 5.3E-12 *exp(190. / temp_loc);
        k_ClO_ClO = k_3rd(temp_loc , cair_loc , 1.9E-32 , 3.6 , 3.7E-12 , 1.6 , 0.6);
        k_BrO_NO2 = k_3rd_iupac(temp_loc , cair_loc , 4.7E-31 , 3.1 , 1.8E-11 , 0.0 , 0.4);
        k_I_NO2 = k_3rd_iupac(temp_loc , cair_loc , 3.0E-31 , 1.0 , 6.6E-11 , 0.0 , 0.63);
        k_DMS_OH = 1.E-9 *exp(5820. / temp_loc) *var[ind_O2] / (1.E30+ 5. *exp(6280. / temp_loc) *var[ind_O2]);
        k_CH2OO_SO2 = 3.66E-11;
        k_O3s = (1.7E-12 *exp(- 940. / temp_loc)) *var[ind_OH] + (1.E-14 *exp(- 490. / temp_loc)) *var[ind_HO2] + jx(index,ip_O1D) *2.2E-10 *var[ind_H2O] / (3.2E-11 *exp(70. / temp_loc) *var[ind_O2] + 1.8E-11 *exp(110. / temp_loc) *var[ind_N2] + 2.2E-10 *var[ind_H2O]);
        beta_null_CH3NO3 = 0.00295 + 5.15E-22 *cair_loc * pow(temp_loc / 298, 7.4);
        beta_inf_CH3NO3 = 0.022;
        beta_CH3NO3 = (beta_null_CH3NO3 *beta_inf_CH3NO3) / (beta_null_CH3NO3 + beta_inf_CH3NO3) / 10.;
        k_NO2_CH3O2 = k_3rd(temp_loc , cair_loc , 1.0E-30 , 4.8 , 7.2E-12 , 2.1 , 0.6);
        k_C6H5O2_NO2 = k_NO2_CH3O2;
        k_CH2OO_NO2 = 4.25E-12;
        beta_C2H5NO3 = (1- 1 / (1+ 1.E-2 *(3.88e-3 *cair_loc / 2.46e19 *760.+ .365) *(1+ 1500. *(1 / temp_loc - 1 / 298.))));
        alpha_NO_HO2 = var[ind_H2O] *6.6E-27 *temp_loc *exp(3700. / temp_loc);
        beta_NO_HO2 = max(((530. / temp_loc)+ (press_loc *4.8004E-6)- 1.73) *0.01 , 0.);
        k0_NO_HO2 = 3.5E-12 *exp(250. / temp_loc);
        k2d_NO_HO2 = (beta_NO_HO2 *k0_NO_HO2) / (1.+ beta_NO_HO2);
        k1d_NO_HO2 = k0_NO_HO2 - k2d_NO_HO2;
        k2w_NO_HO2 = (beta_NO_HO2 *k0_NO_HO2 *(1.+ 42. *alpha_NO_HO2))/ ((1.+ alpha_NO_HO2) *(1.+ beta_NO_HO2));
        k1w_NO_HO2 = k0_NO_HO2 - k2w_NO_HO2;
        k_PrO2_HO2 = 1.9E-13 *exp(1300. / temp_loc);
        k_PrO2_NO = 2.7E-12 *exp(360. / temp_loc);
        k_PrO2_CH3O2 = 9.46E-14 *exp(431. / temp_loc);
        G7402a_yield = 0.8 / 1.1;
        k_CH3CO3_NO2 = k_3rd(temp_loc , cair_loc , 9.7E-29 , 5.6 , 9.3E-12 , 1.5 , 0.6);
        k_PAN_M = k_CH3CO3_NO2 / (9.0E-29 *exp(14000. / temp_loc));
        KRO2NO = 2.54E-12 *exp(360. / temp_loc);

        /*KRO2HO2(:) = 2.91E-13 *exp(1300. / temp_loc) *(1.- exp(- 0.245 *(nC(:))));*/
        for (int ii=0;ii<12;ii++) {
            KRO2HO2[ii] = 2.91E-13 *exp(1300. / temp_loc) * (1.- exp(- 0.245 *float(ii+1)));
        }

        KAPNO = 8.10E-12 *exp(270. / temp_loc);
        KRO2NO3 = 2.50E-12;
        KNO3AL = 1.4E-12 *exp(- 1900. / temp_loc);
        KAPHO2 = 5.20E-13 *exp(980. / temp_loc) *1.865;
        k_CH3O2 = 1.03E-13 *exp(365. / temp_loc);
        k_RO2RCO3 = 2. *2.E-12 *exp(500. / temp_loc);
        k_RO2pRO2 = 2. * pow(1.E-12 *k_CH3O2, .5);
        k_RO2sRO2 = 2. * pow(1.6E-12 *exp(- 2200. / temp_loc) *k_CH3O2, .5);
        k_RO2tRO2 = 2. *3.8E-13 *exp(- 1430. / temp_loc);
        k_RO2pORO2 = 2. *7.5E-13 *exp(500. / temp_loc);
        k_RO2sORO2 = 2. * pow(7.7E-15 *exp(1330. / temp_loc) *k_CH3O2, .5);
        k_RO2tORO2 = 2. * pow(4.7E-13 *exp(- 1420. / temp_loc) *k_CH3O2, .5);
        k_RO2LISOPACO2 = 2. * pow((2.8E-12+ 3.9E-12) / 2. *k_CH3O2, .5);
        k_RO2ISOPBO2 = 2. * pow(6.9E-14 *k_CH3O2, .5);
        k_RO2ISOPDO2 = 2. * pow(4.8E-12 *k_CH3O2, .5);
        k_p = 4.49E-18 *temp_loc *temp_loc *exp(- 320. / temp_loc);
        k_s = 4.50E-18 *temp_loc *temp_loc *exp(253. / temp_loc);
        k_t = 2.12E-18 *temp_loc *temp_loc *exp(696. / temp_loc);
        k_rohro = 2.1E-18 *temp_loc *temp_loc *exp(- 85. / temp_loc);
        k_co2h = .7 *4.2E-14 *exp(850. / temp_loc);
        k_adp = 4.5E-12 * pow(temp_loc / 300., - 0.85);
        k_ads = .25 *(1.1E-11 *exp(485. / temp_loc)+ 1.0E-11 *exp(553. / temp_loc));
        k_adt = 1.922E-11 *exp(450. / temp_loc) - k_ads;
        KHSB = 1.52E11 *exp(- 9512. / temp_loc) *1.;
        KHSD = 6.08E10 *exp(- 8893. / temp_loc) *1.;
        K16HSZ14 = 2.28E9 *exp(- 6764 / temp_loc) *0.28;
        K16HSZ41 = 1.23E9 *exp(- 6186 / temp_loc) *0.28;
        K16HS = pow(K16HSZ14 *K16HSZ41, .5);
        K15HSDHB = 5.;
        K14HSAL = 2.9E7 *exp(- 1 *(5297+ 705) / temp_loc);
        K15HS24VYNAL = K16HSZ14 *exp(- 3500 / (1.987 *temp_loc));
        K15HS42VYNAL = K16HSZ41 *exp(- 3500 / (1.987 *temp_loc));
        KHYDEC = 6.e14 *exp(- 16000. / (1.98588 *temp_loc));
        k_CH2CHOH_OH_HCOOH = 4.3E-11;
        k_CH2CHOH_OH_ALD = 2.4E-11;
        k_CH2CHOH_HCOOH = 4.67E-26 * pow(temp_loc, 3.286 *exp(4509. / (1.987 *temp_loc)));
        k_ALD_HCOOH = 1.17E-19 * pow(temp_loc, 1.209 *exp(- 556. / (1.987 *temp_loc)));
        J_IC3H7NO3 = 3.7 *jx(index,ip_PAN);
        J_ACETOL = 0.65 *0.11 *jx(index,ip_CHOH);
        J_HPALD = (jx(index,ip_CH3OOH)+ jx(index,ip_MACR) / (2. *1.95E-3));
        J_KETENE = jx(index,ip_MVK) / 0.004;
        RO2 = 0.;
        if (ind_LISOPACO2>0) RO2 = RO2 + var[ind_LISOPACO2];
        if (ind_LDISOPACO2>0) RO2 = RO2 + var[ind_LDISOPACO2];
        if (ind_ISOPBO2>0) RO2 = RO2 + var[ind_ISOPBO2];
        if (ind_ISOPDO2>0) RO2 = RO2 + var[ind_ISOPDO2];
        if (ind_LISOPEFO2>0) RO2 = RO2 + var[ind_LISOPEFO2];
        if (ind_NISOPO2>0) RO2 = RO2 + var[ind_NISOPO2];
        if (ind_LHC4ACCO3>0) RO2 = RO2 + var[ind_LHC4ACCO3];
        if (ind_LC578O2>0) RO2 = RO2 + var[ind_LC578O2];
        if (ind_C59O2>0) RO2 = RO2 + var[ind_C59O2];
        if (ind_LNISO3>0) RO2 = RO2 + var[ind_LNISO3];
        if (ind_CH3O2>0) RO2 = RO2 + var[ind_CH3O2];
        if (ind_HOCH2O2>0) RO2 = RO2 + var[ind_HOCH2O2];
        if (ind_CH3CO3>0) RO2 = RO2 + var[ind_CH3CO3];
        if (ind_C2H5O2>0) RO2 = RO2 + var[ind_C2H5O2];
        if (ind_HOCH2CO3>0) RO2 = RO2 + var[ind_HOCH2CO3];
        if (ind_HYPROPO2>0) RO2 = RO2 + var[ind_HYPROPO2];
        if (ind_LBUT1ENO2>0) RO2 = RO2 + var[ind_LBUT1ENO2];
        if (ind_BUT2OLO2>0) RO2 = RO2 + var[ind_BUT2OLO2];
        if (ind_HCOCO3>0) RO2 = RO2 + var[ind_HCOCO3];
        if (ind_CO2H3CO3>0) RO2 = RO2 + var[ind_CO2H3CO3];
        if (ind_LHMVKABO2>0) RO2 = RO2 + var[ind_LHMVKABO2];
        if (ind_MACO3>0) RO2 = RO2 + var[ind_MACO3];
        if (ind_MACRO2>0) RO2 = RO2 + var[ind_MACRO2];
        if (ind_PRONO3BO2>0) RO2 = RO2 + var[ind_PRONO3BO2];
        if (ind_HOCH2CH2O2>0) RO2 = RO2 + var[ind_HOCH2CH2O2];
        if (ind_CH3COCH2O2>0) RO2 = RO2 + var[ind_CH3COCH2O2];
        if (ind_IC3H7O2>0) RO2 = RO2 + var[ind_IC3H7O2];
        if (ind_NC3H7O2>0) RO2 = RO2 + var[ind_NC3H7O2];
        if (ind_LC4H9O2>0) RO2 = RO2 + var[ind_LC4H9O2];
        if (ind_TC4H9O2>0) RO2 = RO2 + var[ind_TC4H9O2];
        if (ind_LMEKO2>0) RO2 = RO2 + var[ind_LMEKO2];
        if (ind_HCOCH2O2>0) RO2 = RO2 + var[ind_HCOCH2O2];
        if (ind_EZCH3CO2CHCHO>0) RO2 = RO2 + var[ind_EZCH3CO2CHCHO];
        if (ind_EZCHOCCH3CHO2>0) RO2 = RO2 + var[ind_EZCHOCCH3CHO2];
        if (ind_CH3COCHO2CHO>0) RO2 = RO2 + var[ind_CH3COCHO2CHO];
        if (ind_HCOCO2CH3CHO>0) RO2 = RO2 + var[ind_HCOCO2CH3CHO];
        if (ind_C1ODC3O2C4OOH>0) RO2 = RO2 + var[ind_C1ODC3O2C4OOH];
        if (ind_C1OOHC2O2C4OD>0) RO2 = RO2 + var[ind_C1OOHC2O2C4OD];
        if (ind_C1ODC2O2C4OD>0) RO2 = RO2 + var[ind_C1ODC2O2C4OD];
        if (ind_ISOPBDNO3O2>0) RO2 = RO2 + var[ind_ISOPBDNO3O2];
        if (ind_LISOPACNO3O2>0) RO2 = RO2 + var[ind_LISOPACNO3O2];
        if (ind_DB1O2>0) RO2 = RO2 + var[ind_DB1O2];
        if (ind_DB2O2>0) RO2 = RO2 + var[ind_DB2O2];
        if (ind_LME3FURANO2>0) RO2 = RO2 + var[ind_LME3FURANO2];
        if (ind_NO3CH2CO3>0) RO2 = RO2 + var[ind_NO3CH2CO3];
        if (ind_CH3COCO3>0) RO2 = RO2 + var[ind_CH3COCO3];
        if (ind_ZCO3C23DBCOD>0) RO2 = RO2 + var[ind_ZCO3C23DBCOD];
        if (ind_IBUTOLBO2>0) RO2 = RO2 + var[ind_IBUTOLBO2];
        if (ind_IPRCO3>0) RO2 = RO2 + var[ind_IPRCO3];
        if (ind_IC4H9O2>0) RO2 = RO2 + var[ind_IC4H9O2];
        if (ind_LMBOABO2>0) RO2 = RO2 + var[ind_LMBOABO2];
        if (ind_IPRHOCO3>0) RO2 = RO2 + var[ind_IPRHOCO3];
        if (ind_LNMBOABO2>0) RO2 = RO2 + var[ind_LNMBOABO2];
        if (ind_NC4OHCO3>0) RO2 = RO2 + var[ind_NC4OHCO3];
        if (ind_LAPINABO2>0) RO2 = RO2 + var[ind_LAPINABO2];
        if (ind_C96O2>0) RO2 = RO2 + var[ind_C96O2];
        if (ind_C97O2>0) RO2 = RO2 + var[ind_C97O2];
        if (ind_C98O2>0) RO2 = RO2 + var[ind_C98O2];
        if (ind_C85O2>0) RO2 = RO2 + var[ind_C85O2];
        if (ind_C86O2>0) RO2 = RO2 + var[ind_C86O2];
        if (ind_PINALO2>0) RO2 = RO2 + var[ind_PINALO2];
        if (ind_C96CO3>0) RO2 = RO2 + var[ind_C96CO3];
        if (ind_C89CO3>0) RO2 = RO2 + var[ind_C89CO3];
        if (ind_C85CO3>0) RO2 = RO2 + var[ind_C85CO3];
        if (ind_OHMENTHEN6ONEO2>0) RO2 = RO2 + var[ind_OHMENTHEN6ONEO2];
        if (ind_C511O2>0) RO2 = RO2 + var[ind_C511O2];
        if (ind_C106O2>0) RO2 = RO2 + var[ind_C106O2];
        if (ind_CO235C6CO3>0) RO2 = RO2 + var[ind_CO235C6CO3];
        if (ind_CHOC3COCO3>0) RO2 = RO2 + var[ind_CHOC3COCO3];
        if (ind_CO235C6O2>0) RO2 = RO2 + var[ind_CO235C6O2];
        if (ind_C716O2>0) RO2 = RO2 + var[ind_C716O2];
        if (ind_C614O2>0) RO2 = RO2 + var[ind_C614O2];
        if (ind_HCOCH2CO3>0) RO2 = RO2 + var[ind_HCOCH2CO3];
        if (ind_BIACETO2>0) RO2 = RO2 + var[ind_BIACETO2];
        if (ind_CO23C4CO3>0) RO2 = RO2 + var[ind_CO23C4CO3];
        if (ind_C109O2>0) RO2 = RO2 + var[ind_C109O2];
        if (ind_C811CO3>0) RO2 = RO2 + var[ind_C811CO3];
        if (ind_C89O2>0) RO2 = RO2 + var[ind_C89O2];
        if (ind_C812O2>0) RO2 = RO2 + var[ind_C812O2];
        if (ind_C813O2>0) RO2 = RO2 + var[ind_C813O2];
        if (ind_C721CO3>0) RO2 = RO2 + var[ind_C721CO3];
        if (ind_C721O2>0) RO2 = RO2 + var[ind_C721O2];
        if (ind_C722O2>0) RO2 = RO2 + var[ind_C722O2];
        if (ind_C44O2>0) RO2 = RO2 + var[ind_C44O2];
        if (ind_C512O2>0) RO2 = RO2 + var[ind_C512O2];
        if (ind_C513O2>0) RO2 = RO2 + var[ind_C513O2];
        if (ind_CHOC3COO2>0) RO2 = RO2 + var[ind_CHOC3COO2];
        if (ind_C312COCO3>0) RO2 = RO2 + var[ind_C312COCO3];
        if (ind_HOC2H4CO3>0) RO2 = RO2 + var[ind_HOC2H4CO3];
        if (ind_LNAPINABO2>0) RO2 = RO2 + var[ind_LNAPINABO2];
        if (ind_C810O2>0) RO2 = RO2 + var[ind_C810O2];
        if (ind_C514O2>0) RO2 = RO2 + var[ind_C514O2];
        if (ind_CHOCOCH2O2>0) RO2 = RO2 + var[ind_CHOCOCH2O2];
        if (ind_ROO6R1O2>0) RO2 = RO2 + var[ind_ROO6R1O2];
        if (ind_ROO6R3O2>0) RO2 = RO2 + var[ind_ROO6R3O2];
        if (ind_RO6R1O2>0) RO2 = RO2 + var[ind_RO6R1O2];
        if (ind_RO6R3O2>0) RO2 = RO2 + var[ind_RO6R3O2];
        if (ind_BPINAO2>0) RO2 = RO2 + var[ind_BPINAO2];
        if (ind_C8BCO2>0) RO2 = RO2 + var[ind_C8BCO2];
        if (ind_NOPINDO2>0) RO2 = RO2 + var[ind_NOPINDO2];
        if (ind_LNBPINABO2>0) RO2 = RO2 + var[ind_LNBPINABO2];
        if (ind_BZBIPERO2>0) RO2 = RO2 + var[ind_BZBIPERO2];
        if (ind_C6H5CH2O2>0) RO2 = RO2 + var[ind_C6H5CH2O2];
        if (ind_TLBIPERO2>0) RO2 = RO2 + var[ind_TLBIPERO2];
        if (ind_BZEMUCCO3>0) RO2 = RO2 + var[ind_BZEMUCCO3];
        if (ind_BZEMUCO2>0) RO2 = RO2 + var[ind_BZEMUCO2];
        if (ind_C5DIALO2>0) RO2 = RO2 + var[ind_C5DIALO2];
        if (ind_NPHENO2>0) RO2 = RO2 + var[ind_NPHENO2];
        if (ind_PHENO2>0) RO2 = RO2 + var[ind_PHENO2];
        if (ind_CRESO2>0) RO2 = RO2 + var[ind_CRESO2];
        if (ind_NCRESO2>0) RO2 = RO2 + var[ind_NCRESO2];
        if (ind_TLEMUCCO3>0) RO2 = RO2 + var[ind_TLEMUCCO3];
        if (ind_TLEMUCO2>0) RO2 = RO2 + var[ind_TLEMUCO2];
        if (ind_C615CO2O2>0) RO2 = RO2 + var[ind_C615CO2O2];
        if (ind_MALDIALCO3>0) RO2 = RO2 + var[ind_MALDIALCO3];
        if (ind_EPXDLCO3>0) RO2 = RO2 + var[ind_EPXDLCO3];
        if (ind_C3DIALO2>0) RO2 = RO2 + var[ind_C3DIALO2];
        if (ind_MALDIALO2>0) RO2 = RO2 + var[ind_MALDIALO2];
        if (ind_C6H5O2>0) RO2 = RO2 + var[ind_C6H5O2];
        if (ind_C6H5CO3>0) RO2 = RO2 + var[ind_C6H5CO3];
        if (ind_OXYL1O2>0) RO2 = RO2 + var[ind_OXYL1O2];
        if (ind_C5CO14O2>0) RO2 = RO2 + var[ind_C5CO14O2];
        if (ind_NBZFUO2>0) RO2 = RO2 + var[ind_NBZFUO2];
        if (ind_BZFUO2>0) RO2 = RO2 + var[ind_BZFUO2];
        if (ind_HCOCOHCO3>0) RO2 = RO2 + var[ind_HCOCOHCO3];
        if (ind_CATEC1O2>0) RO2 = RO2 + var[ind_CATEC1O2];
        if (ind_MCATEC1O2>0) RO2 = RO2 + var[ind_MCATEC1O2];
        if (ind_C5DICARBO2>0) RO2 = RO2 + var[ind_C5DICARBO2];
        if (ind_NTLFUO2>0) RO2 = RO2 + var[ind_NTLFUO2];
        if (ind_TLFUO2>0) RO2 = RO2 + var[ind_TLFUO2];
        if (ind_NPHEN1O2>0) RO2 = RO2 + var[ind_NPHEN1O2];
        if (ind_NNCATECO2>0) RO2 = RO2 + var[ind_NNCATECO2];
        if (ind_NCATECO2>0) RO2 = RO2 + var[ind_NCATECO2];
        if (ind_NBZQO2>0) RO2 = RO2 + var[ind_NBZQO2];
        if (ind_PBZQO2>0) RO2 = RO2 + var[ind_PBZQO2];
        if (ind_NPTLQO2>0) RO2 = RO2 + var[ind_NPTLQO2];
        if (ind_PTLQO2>0) RO2 = RO2 + var[ind_PTLQO2];
        if (ind_NCRES1O2>0) RO2 = RO2 + var[ind_NCRES1O2];
        if (ind_MNNCATECO2>0) RO2 = RO2 + var[ind_MNNCATECO2];
        if (ind_MNCATECO2>0) RO2 = RO2 + var[ind_MNCATECO2];
        if (ind_MECOACETO2>0) RO2 = RO2 + var[ind_MECOACETO2];
        if (ind_CO2H3CO3>0) RO2 = RO2 + var[ind_CO2H3CO3];
        if (ind_MALANHYO2>0) RO2 = RO2 + var[ind_MALANHYO2];
        if (ind_NDNPHENO2>0) RO2 = RO2 + var[ind_NDNPHENO2];
        if (ind_DNPHENO2>0) RO2 = RO2 + var[ind_DNPHENO2];
        if (ind_NDNCRESO2>0) RO2 = RO2 + var[ind_NDNCRESO2];
        if (ind_DNCRESO2>0) RO2 = RO2 + var[ind_DNCRESO2];
        if (ind_C5CO2OHCO3>0) RO2 = RO2 + var[ind_C5CO2OHCO3];
        if (ind_C6CO2OHCO3>0) RO2 = RO2 + var[ind_C6CO2OHCO3];
        if (ind_MMALANHYO2>0) RO2 = RO2 + var[ind_MMALANHYO2];
        if (ind_ACCOMECO3>0) RO2 = RO2 + var[ind_ACCOMECO3];
        if (ind_C4CO2DBCO3>0) RO2 = RO2 + var[ind_C4CO2DBCO3];
        if (ind_C5CO2DBCO3>0) RO2 = RO2 + var[ind_C5CO2DBCO3];
        if (ind_NSTYRENO2>0) RO2 = RO2 + var[ind_NSTYRENO2];
        if (ind_STYRENO2>0) RO2 = RO2 + var[ind_STYRENO2];
        k1_RO2RCO3 = RO2 *k_RO2RCO3;
        k1_RO2pRO2 = RO2 *k_RO2pRO2;
        k1_RO2sRO2 = RO2 *k_RO2sRO2;
        k1_RO2tRO2 = RO2 *k_RO2tRO2;
        k1_RO2pORO2 = RO2 *k_RO2pORO2;
        k1_RO2sORO2 = RO2 *k_RO2sORO2;
        k1_RO2tORO2 = RO2 *k_RO2tORO2;
        k1_RO2LISOPACO2 = RO2 *k_RO2LISOPACO2;
        k1_RO2ISOPBO2 = RO2 *k_RO2ISOPBO2;
        k1_RO2ISOPDO2 = RO2 *k_RO2ISOPDO2;

        rconst(index,0) = (3.3E-11 *exp(55. / temp_loc));
        rconst(index,1) = (6.0E-34 *( pow(temp_loc / 300., - 2.4) )*cair_loc);
        rconst(index,3) = (8.0E-12 *exp(- 2060. / temp_loc));
        rconst(index,4) = (k_3rd(temp_loc , cair_loc , 4.4E-32 , 1.3 , 7.5E-11 , - 0.2 , 0.6));
        rconst(index,5) = (1.4E-10 *exp(- 470. / temp_loc));
        rconst(index,7) = (1.8E-11 *exp(180. / temp_loc));
        rconst(index,8) = (1.7E-12 *exp(- 940. / temp_loc));
        rconst(index,9) = (2.8E-12 *exp(- 1800. / temp_loc));
        rconst(index,10) = (3.E-11 *exp(200. / temp_loc));
        rconst(index,11) = (1.E-14 *exp(- 490. / temp_loc));
        rconst(index,15) = (4.8E-11 *exp(250. / temp_loc));
        rconst(index,16) = (k_HO2_HO2);
        rconst(index,17) = (1.63E-10 *exp(60. / temp_loc));
        rconst(index,19) = (1.5E-11 *exp(- 3600. / temp_loc));
        rconst(index,20) = (2.15E-11 *exp(110. / temp_loc));
        rconst(index,21) = (7.259E-11 *exp(20. / temp_loc));
        rconst(index,22) = (4.641E-11 *exp(20. / temp_loc));
        rconst(index,23) = (3.0E-12 *exp(- 1500. / temp_loc));
        rconst(index,24) = (2.1E-11 *exp(100. / temp_loc));
        rconst(index,25) = (5.1E-12 *exp(210. / temp_loc));
        rconst(index,26) = (1.2E-13 *exp(- 2450. / temp_loc));
        rconst(index,27) = (5.8E-12 *exp(220. / temp_loc));
        rconst(index,28) = (1.5E-11 *exp(170. / temp_loc));
        rconst(index,29) = (k_NO3_NO2);
        rconst(index,30) = (k_NO3_NO2 / (5.8E-27 *exp(10840. / temp_loc)));
        rconst(index,31) = (k_3rd(temp_loc , cair_loc , 7.0E-31 , 2.6 , 3.6E-11 , 0.1 , 0.6));
        rconst(index,32) = (3.3E-12 *exp(270. / temp_loc));
        rconst(index,33) = (k_3rd(temp_loc , cair_loc , 1.8E-30 , 3.0 , 2.8E-11 , 0. , 0.6));
        rconst(index,34) = (k_NO2_HO2);
        rconst(index,36) = (1.8E-11 *exp(- 390. / temp_loc));
        rconst(index,37) = (k_HNO3_OH);
        rconst(index,38) = (k_NO2_HO2 / (2.1E-27 *exp(10900. / temp_loc)));
        rconst(index,39) = (1.3E-12 *exp(380. / temp_loc));
        rconst(index,40) = (1.7E-12 *exp(- 710. / temp_loc));
        rconst(index,41) = (4.3E-12 *exp(- 930. / temp_loc));
        rconst(index,42) = (4.8E-07 *exp(- 628. / temp_loc) * pow(temp_loc, - 1.32) );
        rconst(index,43) = (9.4E-09 *exp(- 356. / temp_loc) * pow(temp_loc, - 1.12) );
        rconst(index,44) = (1.92E-12 *( pow(temp_loc / 298., - 1.5) ));
        rconst(index,45) = (1.41E-11 *( pow(temp_loc / 298., - 1.5) ));
        rconst(index,46) = (1.2E-11 *( pow(temp_loc / 298., - 2.0) ));
        rconst(index,47) = (0.8E-11 *( pow(temp_loc / 298., - 2.0) ));
        rconst(index,50) = (8.0E-11 *exp(- 500. / temp_loc));
        rconst(index,51) = (1.66E-12 *exp(- 1500. / temp_loc));
        rconst(index,52) = (1.0E-12 *exp(- 1000. / temp_loc));
        rconst(index,54) = (4.13E-11 *exp(- 2138. / temp_loc));
        rconst(index,55) = (3.65E-14 *exp(- 4600. / temp_loc));
        rconst(index,57) = (1.85E-20 *exp(2.82 *log(temp_loc)- 987. / temp_loc));
        rconst(index,58) = (2.9E-12 *exp(- 345. / temp_loc));
        rconst(index,59) = (4.1E-13 *exp(750. / temp_loc));
        rconst(index,60) = (2.8E-12 *exp(300. / temp_loc));
        rconst(index,62) = (9.5E-14 *exp(390. / temp_loc) / (1.+ 1. / 26.2 *exp(1130. / temp_loc)));
        rconst(index,63) = (9.5E-14 *exp(390. / temp_loc) / (1.+ 26.2 *exp(- 1130. / temp_loc)));
        rconst(index,64) = (k_CH3OOH_OH);
        rconst(index,65) = (9.52E-18 *exp(2.03 *log(temp_loc)+ 636. / temp_loc));
        rconst(index,66) = (3.4E-13 *exp(- 1900. / temp_loc));
        rconst(index,67) = ((1.57E-13+ cair_loc *3.54E-33));
        rconst(index,69) = (1.49E-17 *temp_loc *temp_loc *exp(- 499. / temp_loc));
        rconst(index,70) = (1.2E-14 *exp(- 2630. / temp_loc));
        rconst(index,71) = (k_3rd(temp_loc , cair_loc , 1.0E-28 , 4.5 , 7.5E-12 , 0.85 , 0.6));
        rconst(index,72) = (7.5E-13 *exp(700. / temp_loc));
        rconst(index,73) = (2.6E-12 *exp(365. / temp_loc));
        rconst(index,75) = (1.6E-13 *exp(195. / temp_loc));
        rconst(index,76) = (k_CH3OOH_OH);
        rconst(index,77) = (4.4E-12 *exp(365. / temp_loc));
        rconst(index,78) = (1.4E-12 *exp(- 1900. / temp_loc));
        rconst(index,79) = (4.2E-14 *exp(855. / temp_loc));
        rconst(index,80) = (4.3E-13 *exp(1040. / temp_loc) / (1.+ 1. / 37. *exp(660. / temp_loc)));
        rconst(index,81) = (4.3E-13 *exp(1040. / temp_loc) / (1.+ 37. *exp(- 660. / temp_loc)));
        rconst(index,82) = (8.1E-12 *exp(270. / temp_loc));
        rconst(index,83) = (k_CH3CO3_NO2);
        rconst(index,85) = (0.9 *2.0E-12 *exp(500. / temp_loc));
        rconst(index,86) = (0.1 *2.0E-12 *exp(500. / temp_loc));
        rconst(index,87) = (4.9E-12 *exp(211. / temp_loc));
        rconst(index,88) = (2.5E-12 *exp(500. / temp_loc));
        rconst(index,89) = (0.6 *k_CH3OOH_OH);
        rconst(index,90) = (5.6E-12 *exp(270. / temp_loc));
        rconst(index,91) = (9.50E-13 *exp(- 650. / temp_loc));
        rconst(index,92) = (k_PAN_M);
        rconst(index,93) = (1.65E-17 *temp_loc *temp_loc *exp(- 87. / temp_loc));
        rconst(index,94) = (6.5E-15 *exp(- 1900. / temp_loc));
        rconst(index,95) = (k_3rd(temp_loc , cair_loc , 8.E-27 , 3.5 , 3.E-11 , 0. , 0.5));
        rconst(index,96) = (4.6E-13 *exp(- 1155. / temp_loc));
        rconst(index,97) = (k_PrO2_HO2);
        rconst(index,98) = (k_PrO2_NO);
        rconst(index,99) = (k_PrO2_CH3O2);
        rconst(index,100) = (k_CH3OOH_OH);
        rconst(index,101) = (6.5E-13 *exp(650. / temp_loc));
        rconst(index,102) = (4.2E-12 *exp(180. / temp_loc));
        rconst(index,103) = (3.8E-12 *exp(200. / temp_loc));
        rconst(index,104) = (1.33E-13+ 3.82E-11 *exp(- 2000. / temp_loc));
        rconst(index,105) = (8.6E-13 *exp(700. / temp_loc));
        rconst(index,106) = (2.9E-12 *exp(300. / temp_loc));
        rconst(index,107) = (7.5E-13 *exp(500. / temp_loc));
        rconst(index,108) = (k_CH3OOH_OH);
        rconst(index,109) = (2.15E-12 *exp(305. / temp_loc));
        rconst(index,110) = (8.4E-13 *exp(830. / temp_loc));
        rconst(index,111) = (6.2E-13 *exp(- 230. / temp_loc));
        rconst(index,112) = (1.81E-17 *temp_loc *temp_loc *exp(114. / temp_loc));
        rconst(index,113) = (k_PrO2_CH3O2);
        rconst(index,114) = (k_PrO2_HO2);
        rconst(index,115) = (k_PrO2_NO);
        rconst(index,116) = (k_CH3OOH_OH);
        rconst(index,117) = (.5 *(1.36E-15 *exp(- 2112. / temp_loc)+ 7.51E-16 *exp(- 1521. / temp_loc)));
        rconst(index,118) = (.5 *(4.1E-12 *exp(452. / temp_loc)+ 1.9E-11 *exp(175. / temp_loc)));
        rconst(index,119) = (1.82E-13 *exp(1300. / temp_loc));
        rconst(index,120) = (2.54E-12 *exp(360. / temp_loc));
        rconst(index,121) = (.25 *k_3rd(temp_loc , cair_loc , 9.7E-29 , 5.6 , 9.3E-12 , 1.5 , 0.6));
        rconst(index,125) = (1.3E-12 *exp(- 25. / temp_loc));
        rconst(index,126) = (k_PrO2_HO2);
        rconst(index,127) = (k_PrO2_NO);
        rconst(index,128) = (k_CH3OOH_OH);
        rconst(index,131) = (k_PAN_M);
        rconst(index,132) = (7.86E-15 *exp(- 1913. / temp_loc));
        rconst(index,133) = (2.54E-11 *exp(410. / temp_loc));
        rconst(index,134) = (3.03E-12 *exp(- 446. / temp_loc));
        rconst(index,135) = (2.22E-13 *exp(1300. / temp_loc));
        rconst(index,136) = (2.54E-12 *exp(360. / temp_loc));
        rconst(index,141) = (2.8E-11 *exp(- 250. / temp_loc));
        rconst(index,142) = (2.5E-11 *exp(110. / temp_loc));
        rconst(index,143) = (1.0E-12 *exp(- 1590. / temp_loc));
        rconst(index,144) = (3.0E-11 *exp(- 2450. / temp_loc));
        rconst(index,145) = (3.5E-13 *exp(- 1370. / temp_loc));
        rconst(index,146) = (k_ClO_ClO);
        rconst(index,147) = (k_ClO_ClO / (2.16E-27 *exp(8537. / temp_loc)));
        rconst(index,148) = (3.9E-11 *exp(- 2310. / temp_loc));
        rconst(index,149) = (4.4E-11- 7.5E-11 *exp(- 620. / temp_loc));
        rconst(index,150) = (7.5E-11 *exp(- 620. / temp_loc));
        rconst(index,151) = (1.1E-11 *exp(- 980. / temp_loc));
        rconst(index,152) = (7.3E-12 *exp(300. / temp_loc));
        rconst(index,153) = (2.2E-12 *exp(340. / temp_loc));
        rconst(index,154) = (1.7E-12 *exp(- 230. / temp_loc));
        rconst(index,155) = (3.0E-12 *exp(- 500. / temp_loc));
        rconst(index,156) = (6.2E-12 *exp(295. / temp_loc));
        rconst(index,157) = (k_3rd_iupac(temp_loc , cair_loc , 1.6E-31 , 3.4 , 7.E-11 , 0. , 0.4));
        rconst(index,158) = (6.918E-7 *exp(- 10909. / temp_loc) *cair_loc);
        rconst(index,159) = (4.5E-12 *exp(- 900. / temp_loc));
        rconst(index,160) = (6.2E-12 *exp(145. / temp_loc));
        rconst(index,161) = (6.6E-12 *exp(- 1240. / temp_loc));
        rconst(index,162) = (8.1E-11 *exp(- 34. / temp_loc));
        rconst(index,164) = (1.8E-12 *exp(- 600. / temp_loc));
        rconst(index,167) = (1.96E-12 *exp(- 1200. / temp_loc));
        rconst(index,169) = (1.64E-12 *exp(- 1520. / temp_loc));
        rconst(index,170) = (k_3rd_iupac(temp_loc , cair_loc , 1.85E-29 , 3.3 , 6.0E-10 , 0.0 , 0.4));
        rconst(index,174) = (1.7E-11 *exp(- 800. / temp_loc));
        rconst(index,175) = (1.9E-11 *exp(230. / temp_loc));
        rconst(index,177) = (2.9E-14 *exp(840. / temp_loc));
        rconst(index,178) = (7.7E-12 *exp(- 450. / temp_loc));
        rconst(index,179) = (4.5E-12 *exp(500. / temp_loc));
        rconst(index,180) = (6.7E-12 *exp(155. / temp_loc));
        rconst(index,181) = (1.2E-10 *exp(- 430. / temp_loc));
        rconst(index,182) = (2.0E-11 *exp(240. / temp_loc));
        rconst(index,184) = (8.7E-12 *exp(260. / temp_loc));
        rconst(index,185) = (k_BrO_NO2);
        rconst(index,186) = (k_BrO_NO2 / (5.44E-9 *exp(14192. / temp_loc) *1.E6 *R_gas *temp_loc / (atm2Pa *N_A)));
        rconst(index,187) = (7.7E-12 *exp(- 580. / temp_loc));
        rconst(index,188) = (2.6E-12 *exp(- 1600. / temp_loc));
        rconst(index,189) = (G7402a_yield *5.7E-12);
        rconst(index,190) = ((1.- G7402a_yield) *5.7E-12);
        rconst(index,191) = (1.42E-12 *exp(- 1150. / temp_loc));
        rconst(index,192) = (2.8E-13 *exp(224. / temp_loc) / (1.+ 1.13E24 *exp(- 3200. / temp_loc) / var[ind_O2]));
        rconst(index,193) = (1.8e-11 *exp(- 460. / temp_loc));
        rconst(index,194) = (9.0E-13 *exp(- 360. / temp_loc));
        rconst(index,195) = (2.0E-12 *exp(- 840. / temp_loc));
        rconst(index,198) = (2.3E-10 *exp(135. / temp_loc));
        rconst(index,199) = (1.6E-12 *exp(430. / temp_loc));
        rconst(index,200) = (2.9E-12 *exp(220. / temp_loc));
        rconst(index,201) = (5.8E-13 *exp(170. / temp_loc));
        rconst(index,203) = (2.0E-12 *exp(- 840. / temp_loc));
        rconst(index,204) = (2.0E-12 *exp(- 840. / temp_loc));
        rconst(index,205) = (2.1E-12 *exp(- 880. / temp_loc));
        rconst(index,206) = (k_3rd(temp_loc , cair_loc , 3.3E-31 , 4.3 , 1.6E-12 , 0. , 0.6));
        rconst(index,207) = (1.13E-11 *exp(- 253. / temp_loc));
        rconst(index,208) = (k_DMS_OH);
        rconst(index,209) = (1.9E-13 *exp(520. / temp_loc));
        rconst(index,211) = (1.8E13 *exp(- 8661. / temp_loc));
        rconst(index,215) = (9.E-11 *exp(- 2386. / temp_loc));
        rconst(index,217) = (jx(index,ip_O2));
        rconst(index,218) = (jx(index,ip_O1D));
        rconst(index,219) = (jx(index,ip_O3P));
        rconst(index,220) = (jx(index,ip_H2O));
        rconst(index,221) = (jx(index,ip_H2O2));
        rconst(index,222) = (jx(index,ip_N2O));
        rconst(index,223) = (jx(index,ip_NO2));
        rconst(index,224) = (jx(index,ip_NO));
        rconst(index,225) = (jx(index,ip_NO2O));
        rconst(index,226) = (jx(index,ip_NOO2));
        rconst(index,227) = (jx(index,ip_N2O5));
        rconst(index,228) = (jx(index,ip_HONO));
        rconst(index,229) = (jx(index,ip_HNO3));
        rconst(index,230) = (jx(index,ip_HNO4));
        rconst(index,231) = (jx(index,ip_CH3OOH));
        rconst(index,232) = (jx(index,ip_COH2));
        rconst(index,233) = (jx(index,ip_CHOH));
        rconst(index,234) = (jx(index,ip_CO2));
        rconst(index,235) = (jx(index,ip_CH4));
        rconst(index,236) = (jx(index,ip_CH3OOH));
        rconst(index,237) = (jx(index,ip_CH3CHO));
        rconst(index,238) = (jx(index,ip_CH3CO3H));
        rconst(index,239) = (0.19 *jx(index,ip_CHOH));
        rconst(index,240) = (jx(index,ip_PAN));
        rconst(index,241) = (jx(index,ip_CH3OOH));
        rconst(index,242) = (jx(index,ip_CH3COCH3));
        rconst(index,243) = (0.074 *jx(index,ip_CHOH));
        rconst(index,244) = (jx(index,ip_MGLYOX));
        rconst(index,245) = (jx(index,ip_CH3OOH));
        rconst(index,246) = (3.7 *jx(index,ip_PAN));
        rconst(index,247) = (jx(index,ip_CH3OOH));
        rconst(index,248) = (0.019 *jx(index,ip_COH2)+ .015 *jx(index,ip_MGLYOX));
        rconst(index,249) = (jx(index,ip_CH3OOH));
        rconst(index,250) = (0.42 *jx(index,ip_CHOH));
        rconst(index,251) = (jx(index,ip_CH3OOH));
        rconst(index,252) = (2.15 *jx(index,ip_MGLYOX));
        rconst(index,253) = (3.7 *jx(index,ip_PAN));
        rconst(index,254) = (jx(index,ip_PAN));
        rconst(index,255) = (jx(index,ip_CH3OOH));
        rconst(index,256) = (3.7 *jx(index,ip_PAN));
        rconst(index,257) = (jx(index,ip_Cl2));
        rconst(index,258) = (jx(index,ip_Cl2O2));
        rconst(index,259) = (jx(index,ip_OClO));
        rconst(index,260) = (jx(index,ip_HCl));
        rconst(index,261) = (jx(index,ip_HOCl));
        rconst(index,262) = (jx(index,ip_ClNO2));
        rconst(index,263) = (jx(index,ip_ClNO3));
        rconst(index,264) = (jx(index,ip_ClONO2));
        rconst(index,265) = (jx(index,ip_CH3Cl));
        rconst(index,266) = (jx(index,ip_CCl4));
        rconst(index,267) = (jx(index,ip_CH3CCl3));
        rconst(index,268) = (jx(index,ip_CFCl3));
        rconst(index,269) = (jx(index,ip_CF2Cl2));
        rconst(index,270) = (jx(index,ip_Br2));
        rconst(index,271) = (jx(index,ip_BrO));
        rconst(index,272) = (jx(index,ip_HOBr));
        rconst(index,273) = (jx(index,ip_BrNO2));
        rconst(index,274) = (jx(index,ip_BrNO3));
        rconst(index,275) = (jx(index,ip_CH3Br));
        rconst(index,276) = (jx(index,ip_CH2Br2));
        rconst(index,277) = (jx(index,ip_CHBr3));
        rconst(index,278) = (jx(index,ip_CF3Br));
        rconst(index,279) = (jx(index,ip_BrCl));
        rconst(index,280) = (jx(index,ip_CF2ClBr));
        rconst(index,281) = (jx(index,ip_CH2ClBr));
        rconst(index,282) = (jx(index,ip_CHCl2Br));
        rconst(index,283) = (jx(index,ip_CHClBr2));
        rconst(index,284) = (jx(index,ip_CH3I));
        rconst(index,285) = (khet_st(index,ihs_N2O5_H2O));
        rconst(index,286) = (khet_tr(index,iht_N2O5));
        rconst(index,287) = (khet_st(index,ihs_HOCl_HCl));
        rconst(index,288) = (khet_st(index,ihs_ClNO3_HCl));
        rconst(index,289) = (khet_st(index,ihs_ClNO3_H2O));
        rconst(index,290) = (khet_st(index,ihs_N2O5_HCl));
        rconst(index,291) = (khet_st(index,ihs_HOBr_HBr));
        rconst(index,292) = (khet_st(index,ihs_BrNO3_H2O));
        rconst(index,293) = (khet_st(index,ihs_ClNO3_HBr));
        rconst(index,294) = (khet_st(index,ihs_BrNO3_HCl));
        rconst(index,295) = (khet_st(index,ihs_HOCl_HBr));
        rconst(index,296) = (khet_st(index,ihs_HOBr_HCl));
        rconst(index,297) = (k_O3s);
        rconst(index,299) = (jx(index,ip_CFCl3));
        rconst(index,301) = (jx(index,ip_CF2Cl2));
        rconst(index,302) = (7.25E-11 *exp(20. / temp_loc));
        rconst(index,303) = (4.63E-11 *exp(20. / temp_loc));
        rconst(index,304) = (jx(index,ip_N2O));
        rconst(index,306) = (1.64E-12 *exp(- 1520. / temp_loc));
        rconst(index,307) = (jx(index,ip_CH3CCl3));
        rconst(index,308) = (jx(index,ip_CF2ClBr));
        rconst(index,309) = (jx(index,ip_CF3Br));
        rconst(index,(3)-1) = 1.2e-10;
        rconst(index,(7)-1) = 1.2e-10;
        rconst(index,(13)-1) = 7.2e-11;
        rconst(index,(14)-1) = 6.9e-12;
        rconst(index,(15)-1) = 1.6e-12;
        rconst(index,(19)-1) = 1.8e-12;
        rconst(index,(36)-1) = 3.5e-12;
        rconst(index,(49)-1) = 1.2e-14;
        rconst(index,(50)-1) = 1300;
        rconst(index,(54)-1) = 1.66e-12;
        rconst(index,(57)-1) = 1.75e-10;
        rconst(index,(62)-1) = 1.3e-12;
        rconst(index,(69)-1) = 4e-13;
        rconst(index,(75)-1) = 2.3e-12;
        rconst(index,(85)-1) = 4e-12;
        rconst(index,(123)-1) = 2e-12;
        rconst(index,(124)-1) = 2e-12;
        rconst(index,(125)-1) = 3e-11;
        rconst(index,(130)-1) = 1.7e-12;
        rconst(index,(131)-1) = 3.2e-11;
        rconst(index,(138)-1) = 2e-12;
        rconst(index,(139)-1) = 2e-12;
        rconst(index,(140)-1) = 1e-10;
        rconst(index,(141)-1) = 1.3e-11;
        rconst(index,(164)-1) = 5.9e-11;
        rconst(index,(166)-1) = 3.3e-10;
        rconst(index,(167)-1) = 1.65e-10;
        rconst(index,(169)-1) = 3.25e-10;
        rconst(index,(172)-1) = 8e-11;
        rconst(index,(173)-1) = 1.4e-10;
        rconst(index,(174)-1) = 2.3e-10;
        rconst(index,(177)-1) = 2.7e-12;
        rconst(index,(184)-1) = 4.9e-11;
        rconst(index,(197)-1) = 3.32e-15;
        rconst(index,(198)-1) = 1.1e-15;
        rconst(index,(203)-1) = 1.45e-11;
        rconst(index,(211)-1) = 1e-10;
        rconst(index,(213)-1) = 3e-13;
        rconst(index,(214)-1) = 5e-11;
        rconst(index,(215)-1) = 3.3e-10;
        rconst(index,(217)-1) = 4.4e-13;
        rconst(index,(299)-1) = 2.3e-10;
        rconst(index,(301)-1) = 1.4e-10;
        rconst(index,(306)-1) = 3e-10;
    }
}


__global__ 
void Rosenbrock(double * __restrict__ conc, const double Tstart, const double Tend, double * __restrict__ rstatus, int * __restrict__ istatus,
                // values calculated from icntrl and rcntrl at host
                const int autonomous, const int vectorTol, const int UplimTol, const int method, const int Max_no_steps,
                const double Hmin, const double Hmax, const double Hstart, const double FacMin, const double FacMax, const double FacRej, const double FacSafe, const double roundoff,
                // cuda global mem buffers              
                const double * __restrict__ absTol, const double * __restrict__ relTol,
                // for update_rconst
    	        const double * __restrict__ khet_st, const double * __restrict__ khet_tr,
		const double * __restrict__ jx,
                // global input
                const double * __restrict__ temp_gpu,
                const double * __restrict__ press_gpu,
                const double * __restrict__ cair_gpu,
                // extra
                const int VL_GLO)
{
    int index = blockIdx.x*blockDim.x+threadIdx.x;

    /* Temporary arrays allocated in stack */

    /* 
     *  Optimization NOTE: runs faster on Tesla/Fermi 
     *  when tempallocated on stack instead of heap.
     *  In theory someone can aggregate accesses together,
     *  however due to algorithm, threads access 
     *  different parts of memory, making it harder to
     *  optimize accesses. 
     *
     */
    double varNew_stack[NVAR];
    double var_stack[NSPEC];
    double varErr_stack[NVAR];
    double fix_stack[NFIX];
    double Fcn0_stack[NVAR];
    double jac0_stack[LU_NONZERO];
    double dFdT_stack[NVAR];
    double Ghimj_stack[LU_NONZERO];
    double K_stack[6*NVAR];


    /* Allocated in Global mem */
    double rconst_stack[NREACT];

    /* Allocated in stack */
    double *Ghimj  = Ghimj_stack;
    double *K      = K_stack;
    double *varNew = varNew_stack;
    double *Fcn0   = Fcn0_stack;
    double *dFdT   = dFdT_stack;
    double *jac0   = jac0_stack;
    double *varErr = varErr_stack;
    double *var    = var_stack;
    double *fix    = fix_stack;  
    double *rconst = rconst_stack;

    if (index < VL_GLO)
    {

        int Nfun,Njac,Nstp,Nacc,Nrej,Ndec,Nsol,Nsng;
        double Texit, Hexit;

        Nfun = 0;
        Njac = 0;
        Nstp = 0;
        Nacc = 0;
        Nrej = 0;
        Ndec = 0;
        Nsol = 0;
        Nsng = 0;

        /* FIXME: add check for method */
        const double *ros_A     = &ros[method-1].ros_A[0]; 
        const double *ros_C     = &ros[method-1].ros_C[0];
        const double *ros_M     = &ros[method-1].ros_M[0]; 
        const double *ros_E     = &ros[method-1].ros_E[0];
        const double *ros_Alpha = &ros[method-1].ros_Alpha[0]; 
        const double *ros_Gamma = &ros[method-1].ros_Gamma[0]; 
        const int    *ros_NewF  = &ros[method-1].ros_NewF[0];
        const int     ros_S     =  ros[method-1].ros_S; 
        const double  ros_ELO   =  ros[method-1].ros_ELO; 





        /* Copy data from global memory to temporary array */
        /*
         * Optimization note: if we ever have enough constant
         * memory, we could use it for storing the data.
         * In current architectures if we use constant memory
         * only a few threads will be able to run on the fly.
         *
         */
        for (int i=0; i<NSPEC; i++)
            var(index,i) = conc(index,i);

        for (int i=0; i<NFIX; i++)
            fix(index,i) = conc(index,NVAR+i);


        update_rconst(var, khet_st, khet_tr, jx, rconst, temp_gpu, press_gpu, cair_gpu, VL_GLO); 

        ros_Integrator(var, fix, Tstart, Tend, Texit,
                //  Rosenbrock method coefficients
                ros_S, ros_M, ros_E, ros_A, ros_C, 
                ros_Alpha, ros_Gamma, ros_ELO, ros_NewF, 
                //  Integration parameters
                autonomous, vectorTol, Max_no_steps, 
                roundoff, Hmin, Hmax, Hstart, Hexit, 
                FacMin, FacMax, FacRej, FacSafe,
                //  Status parameters
                Nfun, Njac, Nstp, Nacc, Nrej, Ndec, Nsol, Nsng,
                //  cuda global mem buffers              
                rconst, absTol, relTol, varNew, Fcn0,  
                K, dFdT, jac0, Ghimj,  varErr, 
                // For update rconst
                khet_st, khet_tr, jx,
                VL_GLO
                );

        for (int i=0; i<NVAR; i++)
            conc(index,i) = var(index,i); 


        /* Statistics */
        istatus(index,ifun) = Nfun;
        istatus(index,ijac) = Njac;
        istatus(index,istp) = Nstp;
        istatus(index,iacc) = Nacc;
        istatus(index,irej) = Nrej;
        istatus(index,idec) = Ndec;
        istatus(index,isol) = Nsol;
        istatus(index,isng) = Nsng;
        // Last T and H
        rstatus(index,itexit) = Texit;
        rstatus(index,ihexit) = Hexit; 
    }
}



__device__ static int ros_Integrator_ros3(double * __restrict__ var, const double * __restrict__ fix, const double Tstart, const double Tend, double &T,
        //  Integration parameters
        const int autonomous, const int vectorTol, const int Max_no_steps, 
        const double roundoff, const double Hmin, const double Hmax, const double Hstart, double &Hexit, 
        const double FacMin, const double FacMax, const double FacRej, const double FacSafe, 
        //  Status parameters
        int &Nfun, int &Njac, int &Nstp, int &Nacc, int &Nrej, int &Ndec, int &Nsol, int &Nsng,
        //  cuda global mem buffers              
        const double * __restrict__ rconst,  const double * __restrict__ absTol, const double * __restrict__ relTol, double * __restrict__ varNew, double * __restrict__ Fcn0, 
        double * __restrict__ K, double * __restrict__ dFdT, double * __restrict__ jac0, double * __restrict__ Ghimj, double * __restrict__ varErr,
        // for update_rconst
        const double * __restrict__ khet_st, const double * __restrict__ khet_tr,
        const double * __restrict__ jx,
        // VL_GLO
        const int VL_GLO)
{
    int index = blockIdx.x*blockDim.x+threadIdx.x;

    double H, Hnew, HC, HC0,HC1, HG, Fac; // Tau - not used
    double Err; //*varErr;
    int direction;
    int rejectLastH, rejectMoreH;
    const double DELTAMIN = 1.0E-5;

    const int ros_S = 3;

    //   ~~~>  Initial preparations
    T = Tstart;
    Hexit = 0.0;
    H = fmin(Hstart,Hmax);
    if (fabs(H) <= 10.0*roundoff) 
        H = DELTAMIN;

    if (Tend  >=  Tstart)
    {
        direction = + 1;
    }
    else
    {
        direction = - 1;
    }

    rejectLastH=0;
    rejectMoreH=0;

    // TimeLoop: 
    while((direction > 0) && ((T- Tend)+ roundoff <= ZERO) || (direction < 0) && ((Tend-T)+ roundoff <= ZERO))
    {
        if (Nstp > Max_no_steps) //  Too many steps
            return -6;
        //  Step size too small
        if (H <= roundoff){  //  Step size too small
            //if (((T+ 0.1*H) == T) || (H <= roundoff)) {
            return -7;
        }

        //   ~~~>  Limit H if necessary to avoid going beyond Tend
        Hexit = H;
        H = fmin(H,fabs(Tend-T));

        //   ~~~>   Compute the function at current time
        Fun(var, fix, rconst, Fcn0, Nfun, VL_GLO);

        //   ~~~>  Compute the function derivative with respect to T
        if (!autonomous)
            ros_FunTimeDerivative(T, roundoff, var, fix, rconst, dFdT, Fcn0, Nfun, khet_st, khet_tr, jx,  VL_GLO); /// VAR READ - fcn0 read

        //   ~~~>   Compute the Jacobian at current time
        Jac_sp(var, fix, rconst, jac0, Njac, VL_GLO);   /// VAR READ 

        //   ~~~>  Repeat step calculation until current step accepted
        // UntilAccepted: 
        while(1)
        {
            ros_PrepareMatrix(H, direction, 0.43586652150845899941601945119356E+00 , jac0, Ghimj, Nsng, Ndec, VL_GLO);

            { // istage=0
                for (int i=0; i<NVAR; i++){
                    K(index,0,i)  = Fcn0(index,i);				// FCN0 Read
                }

                if ((!autonomous))
                {
                    HG = direction*H*0.43586652150845899941601945119356E+00;
                    for (int i=0; i<NVAR; i++){
                        K(index,0,i) += dFdT(index,i)*HG;
		     }
                }
                ros_Solve(Ghimj, K, Nsol, 0, ros_S);
            } // Stage

            {   // istage = 1
                for (int i=0; i<NVAR; i++){		
                    varNew(index,i) = K(index,0,i)  + var(index,i);
                }
                Fun(varNew, fix, rconst, varNew, Nfun,VL_GLO); // FCN <- varNew / not overlap 
                HC = -0.10156171083877702091975600115545E+01/(direction*H);
                for (int i=0; i<NVAR; i++){
                    double tmp = K(index,0,i);
                    K(index,1,i) = tmp*HC + varNew(index,i);
                }
                if ((!autonomous))
                {
                    HG = direction*H*0.24291996454816804366592249683314E+00;
                    for (int i=0; i<NVAR; i++){
                        K(index,1,i) += dFdT(index,i)*HG;
		     }
                }
		//	   R   ,RW, RW,  R,        R 
                ros_Solve(Ghimj, K, Nsol, 1, ros_S);
            } // Stage

            {
                int istage = 2;

                HC0 = 0.40759956452537699824805835358067E+01/(direction*H);
                HC1 = 0.92076794298330791242156818474003E+01/(direction*H);

                for (int i=0; i<NVAR; i++){
                    K(index,2,i) = K(index,1,i)*HC1 +   K(index,0,i)*HC0 +  varNew(index,i);
                }
                if ((!autonomous) )
                {
                    HG = direction*H*0.21851380027664058511513169485832E+01;
                    for (int i=0; i<NVAR; i++){
                        K(index,istage,i) += dFdT(index,i)*HG;
		     }
                }
                ros_Solve(Ghimj, K, Nsol, istage, ros_S);
            } // Stage

            //  ~~~>  Compute the new solution
	    for (int i=0; i<NVAR; i++){
                    varNew(index,i) = K(index,0,i)   + K(index,1,i)*0.61697947043828245592553615689730E+01 + K(index,2,i)*(-0.42772256543218573326238373806514) + var(index,i) ;
                    varErr(index,i) = K(index,0,i)/2 + K(index,1,i)*(-0.29079558716805469821718236208017E+01) + K(index,2,i)*(0.22354069897811569627360909276199);
	    }

            Err = ros_ErrorNorm(var, varNew, varErr, absTol, relTol, vectorTol);   

//  ~~~> New step size is bounded by FacMin <= Hnew/H <= FacMax
            Fac  = fmin(FacMax,fmax(FacMin,FacSafe/pow(Err,ONE/3.0)));
            Hnew = H*Fac;

//  ~~~>  Check the error magnitude and adjust step size
            Nstp = Nstp+ 1;
            if((Err <= ONE) || (H <= Hmin)) // ~~~> Accept step
            {
                Nacc = Nacc + 1;
                for (int j=0; j<NVAR ; j++)
                    var(index,j) =  fmax(varNew(index,j),ZERO);  /////////// VAR WRITE - last VarNew read

                T = T +  direction*H;
                Hnew = fmax(Hmin,fmin(Hnew,Hmax));
                if (rejectLastH)   // No step size increase after a rejected step
                    Hnew = fmin(Hnew,H);
                rejectLastH = 0;
                rejectMoreH = 0;
                H = Hnew;

            	break;  //  EXIT THE LOOP: WHILE STEP NOT ACCEPTED
            }
            else      // ~~~> Reject step
            {
                if (rejectMoreH)
                    Hnew = H*FacRej;
                rejectMoreH = rejectLastH;
                rejectLastH = 1;
                H = Hnew;
                if (Nacc >= 1)
                    Nrej += 1;
            } //  Err <= 1
        } // UntilAccepted
    } // TimeLoop
//  ~~~> Succesful exit
    return 0; //  ~~~> The integration was successful
}

__global__ 
void Rosenbrock_ros3(double * __restrict__ conc, const double Tstart, const double Tend, double * __restrict__ rstatus, int * __restrict__ istatus,
                const int autonomous, const int vectorTol, const int UplimTol, const int Max_no_steps,
                const double Hmin, const double Hmax, const double Hstart, const double FacMin, const double FacMax, const double FacRej, const double FacSafe, const double roundoff,
                const double * __restrict__ absTol, const double * __restrict__ relTol,
    	        const double * __restrict__ khet_st, const double * __restrict__ khet_tr,
		const double * __restrict__ jx,
                const double * __restrict__ temp_gpu,
                const double * __restrict__ press_gpu,
                const double * __restrict__ cair_gpu,
                const int VL_GLO)
{
    int index = blockIdx.x*blockDim.x+threadIdx.x;

    /* Temporary arrays allocated in stack */

    /* 
     *  Optimization NOTE: runs faster on Tesla/Fermi 
     *  when tempallocated on stack instead of heap.
     *  In theory someone can aggregate accesses together,
     *  however due to algorithm, threads access 
     *  different parts of memory, making it harder to
     *  optimize accesses. 
     *
     */
    double varNew_stack[NVAR];
    double var_stack[NVAR];
    double varErr_stack[NVAR];
    double fix_stack[NFIX];
    double Fcn0_stack[NVAR];
    double jac0_stack[LU_NONZERO];
    double dFdT_stack[NVAR];
    double Ghimj_stack[LU_NONZERO];
    double K_stack[3*NVAR];
    double rconst_stack[NREACT];

    /* Allocated in stack */
    double *Ghimj  = Ghimj_stack;
    double *K      = K_stack;
    double *varNew = varNew_stack;
    double *Fcn0   = Fcn0_stack;
    double *dFdT   = dFdT_stack;
    double *jac0   = jac0_stack;
    double *varErr = varErr_stack;
    double *var    = var_stack;
    double *fix    = fix_stack;  
    double *rconst = rconst_stack;

    const int method = 2;

    if (index < VL_GLO)
    {

        int Nfun,Njac,Nstp,Nacc,Nrej,Ndec,Nsol,Nsng;
        double Texit, Hexit;

        Nfun = 0;
        Njac = 0;
        Nstp = 0;
        Nacc = 0;
        Nrej = 0;
        Ndec = 0;
        Nsol = 0;
        Nsng = 0;



        /* Copy data from global memory to temporary array */
        /*
         * Optimization note: if we ever have enough constant
         * memory, we could use it for storing the data.
         * In current architectures if we use constant memory
         * only a few threads will be able to run on the fly.
         *
         */
        for (int i=0; i<NSPEC; i++)
            var(index,i) = conc(index,i);

        for (int i=0; i<NFIX; i++)
            fix(index,i) = conc(index,NVAR+i);

        //update_rconst(var, khet_st, khet_tr, jx, VL_GLO);
        update_rconst(var, khet_st, khet_tr, jx, rconst, temp_gpu, press_gpu, cair_gpu, VL_GLO); 

        ros_Integrator_ros3(var, fix, Tstart, Tend, Texit,
                //  Integration parameters
                autonomous, vectorTol, Max_no_steps, 
                roundoff, Hmin, Hmax, Hstart, Hexit, 
                FacMin, FacMax, FacRej, FacSafe,
                //  Status parameters
                Nfun, Njac, Nstp, Nacc, Nrej, Ndec, Nsol, Nsng,
                //  cuda global mem buffers              
                rconst, absTol, relTol, varNew, Fcn0,  
                K, dFdT, jac0, Ghimj,  varErr, 
                // For update rconst
                khet_st, khet_tr, jx,
                VL_GLO
                );

        for (int i=0; i<NVAR; i++)
            conc(index,i) = var(index,i); 


        /* Statistics */
        istatus(index,ifun) = Nfun;
        istatus(index,ijac) = Njac;
        istatus(index,istp) = Nstp;
        istatus(index,iacc) = Nacc;
        istatus(index,irej) = Nrej;
        istatus(index,idec) = Ndec;
        istatus(index,isol) = Nsol;
        istatus(index,isng) = Nsng;
        // Last T and H
        rstatus(index,itexit) = Texit;
        rstatus(index,ihexit) = Hexit; 
    }
}






                                                        // no int8 in CUDA :(
__global__ void reduce_istatus_1(int *istatus, int4 *tmp_out_1, int4 *tmp_out_2, int VL_GLO, int *xNacc, int *xNrej)
{
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    int idx_1 = threadIdx.x;
    int global_size = blockDim.x*gridDim.x;
    
    int foo;
    //no int8 in CUDA :(
    int4 accumulator_1 = make_int4(0,0,0,0);
    int4 accumulator_2 = make_int4(0,0,0,0);
    while (index < VL_GLO)
    {
        accumulator_1.x += istatus(index,0);
        accumulator_1.y += istatus(index,1);
        accumulator_1.z += istatus(index,2);
        //some dirty work on the side...
        foo = istatus(index,3);
        xNacc[index] = foo;
        accumulator_1.w += foo;
        foo = istatus(index,4);
        xNrej[index] = foo;
        accumulator_2.x += foo;
        accumulator_2.y += istatus(index,5);
        accumulator_2.z += istatus(index,6);
        accumulator_2.w += istatus(index,7);
        index += global_size;
    }
    //no int8 in CUDA :(
    __shared__ int4 buffer_1[REDUCTION_SIZE_1];
    __shared__ int4 buffer_2[REDUCTION_SIZE_1];
    
    buffer_1[idx_1] = accumulator_1;
    buffer_2[idx_1] = accumulator_2;
    __syncthreads();
    
    int idx_2, active_threads = blockDim.x;
    int4 tmp_1, tmp_2;
    while (active_threads != 1)
    {
        active_threads /= 2;
        if (idx_1 < active_threads)
        {
            idx_2 = idx_1+active_threads;
            
            tmp_1 = buffer_1[idx_1];
            tmp_2 = buffer_1[idx_2];
            
            tmp_1.x += tmp_2.x;
            tmp_1.y += tmp_2.y;
            tmp_1.z += tmp_2.z;
            tmp_1.w += tmp_2.w;
            
            buffer_1[idx_1] = tmp_1;
            
            
            tmp_1 = buffer_2[idx_1];
            tmp_2 = buffer_2[idx_2];
            
            tmp_1.x += tmp_2.x;
            tmp_1.y += tmp_2.y;
            tmp_1.z += tmp_2.z;
            tmp_1.w += tmp_2.w;
            
            buffer_2[idx_1] = tmp_1;
            
        }
        __syncthreads();
    }
    if (idx_1 == 0)
    {
        tmp_out_1[blockIdx.x] = buffer_1[0];
        tmp_out_2[blockIdx.x] = buffer_2[0];
    }
}            

__global__ void reduce_istatus_2(int4 *tmp_out_1, int4 *tmp_out_2, int *out)
{
    int idx_1 = threadIdx.x;
    //no int8 in CUDA :(
    __shared__ int4 buffer_1[REDUCTION_SIZE_2];
    __shared__ int4 buffer_2[REDUCTION_SIZE_2];
    
    buffer_1[idx_1] = tmp_out_1[idx_1];
    buffer_2[idx_1] = tmp_out_2[idx_1]; 
    __syncthreads();
    
    int idx_2, active_threads = blockDim.x;
    int4 tmp_1, tmp_2;
    while (active_threads != 1)
    {
        active_threads /= 2;
        if (idx_1 < active_threads)
        {
            idx_2 = idx_1+active_threads;
            
            tmp_1 = buffer_1[idx_1];
            tmp_2 = buffer_1[idx_2];
            
            tmp_1.x += tmp_2.x;
            tmp_1.y += tmp_2.y;
            tmp_1.z += tmp_2.z;
            tmp_1.w += tmp_2.w;
            
            buffer_1[idx_1] = tmp_1;
            
            
            tmp_1 = buffer_2[idx_1];
            tmp_2 = buffer_2[idx_2];
            
            tmp_1.x += tmp_2.x;
            tmp_1.y += tmp_2.y;
            tmp_1.z += tmp_2.z;
            tmp_1.w += tmp_2.w;
            
            buffer_2[idx_1] = tmp_1;
            
        }
        __syncthreads();
    }
    if (idx_1 == 0)
    {
        tmp_1 = buffer_1[0];
        tmp_2 = buffer_2[0];
        out[0] = tmp_1.x;
        out[1] = tmp_1.y;
        out[2] = tmp_1.z;
        out[3] = tmp_1.w;
        out[4] = tmp_2.x;
        out[5] = tmp_2.y;
        out[6] = tmp_2.z;
        out[7] = tmp_2.w;        
    }
}            

/* Assuming different processes */
enum { TRUE=1, FALSE=0 } ;
double *d_conc, *d_temp, *d_press, *d_cair, *d_khet_st, *d_khet_tr, *d_jx;
int initialized = FALSE;

/* Device pointers pointing to GPU */
double *d_rstatus, *d_absTol, *d_relTol;
int *d_istatus, *d_istatus_rd, *d_xNacc, *d_xNrej;
int4 *d_tmp_out_1, *d_tmp_out_2;

/* Allocate arrays on device for Rosenbrock */
__host__ void init_first_time(int pe, int VL_GLO, int size_khet_st, int size_khet_tr, int size_jx ){

    /* Select the proper GPU CARD */
    int deviceCount, device;
    gpuErrchk( cudaGetDeviceCount(&deviceCount) );
    device = pe % deviceCount;
    gpuErrchk( cudaSetDevice(device) );

    printf("PE[%d]: selected %d of total %d\n",pe,device,deviceCount);
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1); 

    gpuErrchk( cudaMalloc ((void **) &d_conc   , sizeof(double)*VL_GLO*(NSPEC))        );
    gpuErrchk( cudaMalloc ((void **) &d_khet_st, sizeof(double)*VL_GLO*size_khet_st) );
    gpuErrchk( cudaMalloc ((void **) &d_khet_tr, sizeof(double)*VL_GLO*size_khet_tr) );
    gpuErrchk( cudaMalloc ((void **) &d_jx     , sizeof(double)*VL_GLO*size_jx)      );
   
    gpuErrchk( cudaMalloc ((void **) &d_rstatus    , sizeof(double)*VL_GLO*2)          );
    gpuErrchk( cudaMalloc ((void **) &d_istatus    , sizeof(int)*VL_GLO*8)             );
    gpuErrchk( cudaMalloc ((void **) &d_absTol     , sizeof(double)*NVAR)              );
    gpuErrchk( cudaMalloc ((void **) &d_relTol     , sizeof(double)*NVAR)              );

    /* Allocate input arrays */
    gpuErrchk( cudaMalloc ((void **) &temp_gpu     , sizeof(double)*VL_GLO)              );
    gpuErrchk( cudaMalloc ((void **) &press_gpu     , sizeof(double)*VL_GLO)              );
    gpuErrchk( cudaMalloc ((void **) &cair_gpu     , sizeof(double)*VL_GLO)              );

    /* Allocate arrays on device for reduce_foo */
    gpuErrchk( cudaMalloc ((void **) &d_istatus_rd  , sizeof(int)*8));
    gpuErrchk( cudaMalloc ((void **) &d_tmp_out_1   , sizeof(int4)*64));
    gpuErrchk( cudaMalloc ((void **) &d_tmp_out_2   , sizeof(int4)*64));
    gpuErrchk( cudaMalloc ((void **) &d_xNacc   , sizeof(int)*VL_GLO));
    gpuErrchk( cudaMalloc ((void **) &d_xNrej   , sizeof(int)*VL_GLO));
    

    initialized = TRUE;
}

/*
 * TODO: We should call it in some point..
 */
extern "C" void finalize_cuda(){
    /* Free memory on the device */
    gpuErrchk( cudaFree(d_conc        ) );
    gpuErrchk( cudaFree(d_temp        ) );
    gpuErrchk( cudaFree(d_press       ) );
    gpuErrchk( cudaFree(d_cair        ) );
    gpuErrchk( cudaFree(d_khet_st     ) );
    gpuErrchk( cudaFree(d_khet_tr     ) );
    gpuErrchk( cudaFree(d_jx          ) );
    gpuErrchk( cudaFree(d_rstatus     ) );
    gpuErrchk( cudaFree(d_istatus     ) );
    gpuErrchk( cudaFree(d_absTol      ) );
    gpuErrchk( cudaFree(d_relTol      ) );
    gpuErrchk( cudaFree(d_istatus_rd  ) ); 
    gpuErrchk( cudaFree(d_tmp_out_1   ) ); 
    gpuErrchk( cudaFree(d_tmp_out_2   ) ); 
    gpuErrchk( cudaFree(d_xNacc       ) ); 
    gpuErrchk( cudaFree(d_xNrej       ) ); 
    gpuErrchk( cudaFree(temp_gpu      ) ); 
    gpuErrchk( cudaFree(press_gpu     ) ); 
    gpuErrchk( cudaFree(cair_gpu      ) ); 
}



extern "C" void kpp_integrate_cuda_( int *pe_p, int *sizes, double *time_step_len_p, double *conc, double *temp, double *press, double *cair, 
                                    double *khet_st, double *khet_tr, double *jx, double *absTol, double *relTol, int *ierr, int *istatus, 
                                    int *xNacc, int *xNrej, double *rndoff, int *icntrl=NULL, double *rcntrl=NULL
				    ) 
/*  // TODO
 *  Parameters:
 *          pe_p: scalar int - processor element
 *        VL_GLO: scalar int - size of the system
 *         NSPEC: scalar int - number of species
 *        NREACT: scalar int - number of reactions
 *          NVAR: scalar int - 
 *
 *  Input data:
 *          conc: 2D array of doubles - size: vl_glo x number of species
 *          temp: 1D array of doubles - size: vl_glo
 *         press: 1D array of doubles - size: vl_glo
 *          cair: 1D array of doubles - size: vl_glo
 *       khet_st: 2D array of doubles - size: vl_glo x number of species
 *       khet_tr: 2D array of doubles - size: vl_glo x number of species 
 *            jx: 2D array of doubles - size: vl_glo x number of species
 *        absTol: 1D array of doubles - size: number of species
 *        relTol: 1D array of doubles - size: number of species
 *     Control:
 *        icntrl: 1D array of ints   - size: 4
 *         sizes: 1D array of ints   - size: 4
 *        rcntrl: 1D array of doubles - size: 7
 * 
 * 
 */
{

    const double DELTAMIN = 1.0E-5;


    
    int VL_GLO       = sizes[0];
    int size_khet_st = sizes[1];
    int size_khet_tr = sizes[2];
    int size_jx      = sizes[3];
    double roundoff  = *rndoff; 
    
    double Tstart,Tend;
    Tstart = ZERO;
    Tend   =  *time_step_len_p;
    int pe = *pe_p;
    
    // variables from rcntrl and icntrl
    int autonomous, vectorTol, UplimTol, method, Max_no_steps;
    double Hmin, Hmax, Hstart, FacMin, FacMax, FacRej, FacSafe;
    
    //int rcntrl_bool = 0, icntrl_bool=0;
    if (rcntrl == NULL)
    {
        rcntrl = new double[7];
        for (int i=0; i < 7; i++)
            rcntrl[i] = 0.0;
    }
    if (icntrl == NULL)
    {
        icntrl = new int[4];
        for (int i=0; i < 4; i++)
            icntrl[i] = 0;
    }

    /* Allocate arrays on device for update_rconst kernel*/        
    if (initialized == FALSE)   init_first_time(pe, VL_GLO, size_khet_st, size_khet_tr, size_jx);

    /* Copy data from host memory to device memory */
    gpuErrchk( cudaMemcpy(d_conc   , conc   	, sizeof(double)*VL_GLO*NSPEC        , cudaMemcpyHostToDevice) );

    gpuErrchk( cudaMemcpy(temp_gpu   , temp   	, sizeof(double)*VL_GLO  , cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(press_gpu  , press  	, sizeof(double)*VL_GLO  , cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(cair_gpu   , cair   	, sizeof(double)*VL_GLO  , cudaMemcpyHostToDevice) );

    gpuErrchk( cudaMemcpy(d_khet_st, khet_st	, sizeof(double)*VL_GLO*size_khet_st , cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_khet_tr, khet_tr	, sizeof(double)*VL_GLO*size_khet_tr , cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_jx     , jx     	, sizeof(double)*VL_GLO*size_jx      , cudaMemcpyHostToDevice) );

    /* Copy arrays from host memory to device memory for Rosenbrock */    
    gpuErrchk( cudaMemcpy(d_absTol, absTol, sizeof(double)*NVAR, cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_relTol, relTol, sizeof(double)*NVAR, cudaMemcpyHostToDevice) );


    /* Compute execution configuration for update_rconst */
    int block_size, grid_size;
    
    block_size = BLOCKSIZE;
    grid_size = (VL_GLO + block_size - 1)/block_size;  
    dim3 dimBlock(block_size);
    dim3 dimGrid(grid_size);


    /* Execute the kernel */
    //update_rconst<<<dimGrid,dimBlock>>>(d_conc, d_khet_st, d_khet_tr, d_jx, VL_GLO); 

    GPU_DEBUG();
 
//  *------------------------------------------------------*
//  |    Default values vs input settings (icntrl, rcntrl) |
//  *------------------------------------------------------*
    int ierr_tmp=0;
    {
    //  autonomous or time dependent ODE. Default is time dependent.
        autonomous = !(icntrl[0] == 0);

    //  For Scalar tolerances (icntrl[1].NE.0)  the code uses absTol(0) and relTol(0)
    //  For Vector tolerances (icntrl[1] == 0) the code uses absTol(0:NVAR) and relTol(0:NVAR)
        if (icntrl[1] == 0)
        {
            vectorTol = 1; //bool
            UplimTol  = NVAR;
        }
        else
        {
            vectorTol = 0;
            UplimTol  = 1;
        }

    //  The particular Rosenbrock method chosen
        if (icntrl[2] == 0) 
        {
            method = 4;
        }
        else if ((icntrl[2] >= 1) && (icntrl[2] <= 5))
        {
            method = icntrl[2];
        }
        else
        {
            printf("User-selected Rosenbrock method: icntrl[2]=%d\n",method);
            ierr_tmp = -2;
        }
    //  The maximum number of steps admitted
        if (icntrl[3] == 0)
        {
            Max_no_steps = 100000;
        }
        else if (icntrl[3] > 0) 
        {
            Max_no_steps=icntrl[3];
        }
        else
        {
            printf("User-selected max no. of steps: icntrl[3]=%d\n",icntrl[3]);
            ierr_tmp = -1;
        }
    //  Unit roundoff (1+ roundoff>1)
        roundoff = machine_eps_flt(); 

    //  Lower bound on the step size: (positive value)
        if (rcntrl[0] == ZERO)
        {
            Hmin = ZERO;
        }
        else if (rcntrl[0] > ZERO) 
        {
            Hmin = rcntrl[0];
        }
        else
        {
            printf("User-selected Hmin: rcntrl[0]=%f\n",rcntrl[0]);
            ierr_tmp = -3;
        }
    //  Upper bound on the step size: (positive value)
        if (rcntrl[1] == ZERO) 
        {
            Hmax = fabs(Tend-Tstart);
        }
        else if (rcntrl[1] > ZERO) 
        {
            Hmax = fmin(fabs(rcntrl[1]),fabs(Tend-Tstart));
        }
        else
        {
            printf("User-selected Hmax: rcntrl[1]=%f\n",rcntrl[1]);
            ierr_tmp = -3;
        }
    //  Starting step size: (positive value)
        if (rcntrl[2] == ZERO) 
        {
            Hstart = fmax(Hmin,DELTAMIN);
        }
        else if (rcntrl[2] > ZERO) 
        {
            Hstart = fmin(fabs(rcntrl[2]),fabs(Tend-Tstart));
        }
        else
        {
            printf("User-selected Hstart: rcntrl[2]=%f\n",rcntrl[2]);
            ierr_tmp = -3;
        }
    //  Step size can be changed s.t.  FacMin < Hnew/Hexit < FacMax
        if (rcntrl[3] == ZERO)
        {
            FacMin = 0.2;
        }
        else if (rcntrl[3] > ZERO) 
        {
            FacMin = rcntrl[3];
        }
        else
        {
            printf("User-selected FacMin: rcntrl[3]=%f\n",rcntrl[3]);
            ierr_tmp = -4;
        }
        if (rcntrl[4] == ZERO) 
        {
            FacMax = 6.0;
        }
        else if (rcntrl[4] > ZERO) 
        {
            FacMax = rcntrl[4];
        }
        else
        {
            printf("User-selected FacMax: rcntrl[4]=%f\n",rcntrl[4]);
            ierr_tmp = -4; 
        }
    //  FacRej: Factor to decrease step after 2 succesive rejections
        if (rcntrl[5] == ZERO) 
        {
            FacRej = 0.1;
        }
        else if (rcntrl[5] > ZERO) 
        {
            FacRej = rcntrl[5];
        }
        else
        {
            printf("User-selected FacRej: rcntrl[5]=%f\n",rcntrl[5]);
            ierr_tmp = -4;
        }
    //  FacSafe: Safety Factor in the computation of new step size
        if (rcntrl[6] == ZERO) 
        {
            FacSafe = 0.9;
        }
        else if (rcntrl[6] > ZERO)
        {
            FacSafe = rcntrl[6];
        }
        else
        {
            printf("User-selected FacSafe: rcntrl[6]=%f\n",rcntrl[6]);
            ierr_tmp = -4;
        }
    //  Check if tolerances are reasonable
        for (int i=0; i < UplimTol; i++)
        {
            if ((absTol[i] <= ZERO) || (relTol[i] <= 10.0*roundoff) || (relTol[i] >= 1.0))
            {
                printf("CCC absTol(%d) = %f \n",i,absTol[i]);
                printf("CCC relTol(%d) = %f \n",i,relTol[i]);
                ierr_tmp = -5;
            }
        }
    }


      switch (method){
        case 2:
            Rosenbrock_ros3<<<dimGrid,dimBlock>>>(d_conc, Tstart, Tend, d_rstatus, d_istatus,
                    autonomous, vectorTol, UplimTol, Max_no_steps,
                    Hmin, Hmax, Hstart, FacMin, FacMax, FacRej, FacSafe, roundoff,
                    d_absTol, d_relTol,
                    d_khet_st, d_khet_tr, d_jx, 
                    temp_gpu, press_gpu, cair_gpu, 
                    VL_GLO);
            break;
        default: 
      Rosenbrock<<<dimGrid,dimBlock>>>(d_conc, Tstart, Tend, d_rstatus, d_istatus,
                    // values calculated from icntrl and rcntrl at host
                    autonomous, vectorTol, UplimTol, method, Max_no_steps,
                    Hmin, Hmax, Hstart, FacMin, FacMax, FacRej, FacSafe, roundoff,
                    //  cuda global mem buffers              
                    d_absTol, d_relTol,   
                    d_khet_st, d_khet_tr, d_jx, 
                    // Global input arrays
                    temp_gpu, press_gpu, cair_gpu, 
                    // extra - vector lenght and processor
                    VL_GLO); 
        
                    break;
    }

    GPU_DEBUG();

    
    reduce_istatus_1<<<REDUCTION_SIZE_2,REDUCTION_SIZE_1>>>(d_istatus, d_tmp_out_1, d_tmp_out_2, VL_GLO, d_xNacc, d_xNrej);


    GPU_DEBUG();

    reduce_istatus_2<<<1,REDUCTION_SIZE_2>>>(d_tmp_out_1, d_tmp_out_2, d_istatus_rd);

    GPU_DEBUG();
    
    /* Copy the result back */
    gpuErrchk( cudaMemcpy( conc      , d_conc       , sizeof(double)*VL_GLO*NVAR, cudaMemcpyDeviceToHost) );  
    gpuErrchk( cudaMemcpy( xNacc      , d_xNacc      , sizeof(int)*VL_GLO         , cudaMemcpyDeviceToHost) );  
    gpuErrchk( cudaMemcpy( xNrej      , d_xNrej      , sizeof(int)*VL_GLO         , cudaMemcpyDeviceToHost) ); 

    
    return;

}





