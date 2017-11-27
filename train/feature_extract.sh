#!/bin/sh

awk -F ',' \
'{
	if($8 == "speed_change_video"){	
		if($5 > $4){
			a[$1"#"$2"_"$8"_fast"]+=$5-$4
		}else{
			a[$1"#"$2"_"$8"_slow"]+=$5-$4
		}
	}else if($8 == "seek_video"){
		if($7 > $6){
			a[$1"#"$2"_"$8"_forward"]+=$7-$6
		}else{
			a[$1"#"$2"_"$8"_backward"]+=$7-$6
		}
	}else{
		a[$1"#"$2"_"$8]++
	}
}
END{for (i in a) print i, a[i]} ' ../dataset/TrainFeatures.csv > ../dataset/temp1.csv

sed -i '/user/d' ../dataset/temp1.csv

awk -F ',' '{if(NR == FNR) {if(NR>1)a[$1]=$1} else {for (i in a) {print a[i]"_"$0}} }' ../dataset/VideoInfo.csv ../dataset/event_type.txt > ../dataset/features.csv

sed -i '/user/d' ../dataset/features.csv

awk -F '#' '{a[$1] = a[$1]$2"#"} END{for (i in a) print i"|"a[i]}' ../dataset/temp1.csv > ../dataset/temp2.csv


awk -F '|' \
'{
	if(NR == FNR){
		idx[$1] = NR
	}else{
		a=$1;
		features=""
		for(i=1; i<=504; i++){
			f = 0;
			len = split($2,b,"#");		
			for(j=1;j<=len;j++){
				split(b[j], t, " ");
				k = t[1];
				v = t[2];
				if(idx[k] == i){
					f = v;
					break;
				}
			}
			features = features","f
		}
		print $1features
	}
}' ../dataset/features.csv ../dataset/temp2.csv > ../dataset/temp3.csv

awk -F ',' '{if(NR == FNR){a[$1]=$2}else{print $0","a[$1]}}' ../dataset/TrainLabel.csv ../dataset/temp3.csv > extracted_features.csv

rm ../dataset/temp*










