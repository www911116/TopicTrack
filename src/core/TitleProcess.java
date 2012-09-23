package core;

import java.awt.image.TileObserver;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.math.BigInteger;
import weka.core.*;

public class TitleProcess {
	static int ID;
	/**
	 * @param args
	 */
	
	public TitleProcess() {
		// TODO Auto-generated constructor stub
		ID = 0;
	}
	public String titleStemming(String title){
		String[] titleparts = title.split("_");
		if ( titleparts[0].length() < 4 || titleparts[titleparts.length-1].length() > 5)
			return title;
		else{
			return titleparts[0];
		}
		
	}
	
	public void fileNameChange(File originname, String newname){
		File newfile = new File(originname.getParentFile(),newname);
		originname.renameTo(newfile);
	}
	
	public void renameTitleDir(File dir){
		for(File f:dir.listFiles()){
			if ( f.isDirectory()){
				renameTitleDir(f);
			}else{
				String newtitlename = titleStemming(f.getName());
				File newfile = new File(newtitlename);
				if ( newfile.exists())
					newtitlename = newtitlename + "1";
				fileNameChange(f, newtitlename);
			}
		}		
	}
	
	public void changeStrNameToNum(File dir) throws Exception{
		DataBaseTool db = new DataBaseTool();
		db.init("root","4432519");
		Instances data=new Instances(db.getTableStructure("IDmatchName"));
		for(File f:dir.listFiles()){
			if ( f.isDirectory()){
				changeStrNameToNum(f);
			}else{
				 FileReader fr = new FileReader( f);
				 BufferedReader br = new BufferedReader(fr);
				 StringBuffer txtStr = new StringBuffer();
				 String line=br.readLine();
				 txtStr.append( line + "\n" );
				 br.close();
				 fr.close();
				 double[] newInst = new double[2];
				 newInst[1] = (double) data.attribute(1).addStringValue(txtStr.toString());
				 String newtitlename = String.valueOf(++ID);
				 int t=ID;
				 File newfile = new File(newtitlename);
				 if ( newfile.exists())
					{newtitlename = newtitlename + "123456";t=ID+123456;}
				 fileNameChange(f, newtitlename);
				 newInst[0]=(double) t;
				 Instance a=new Instance(1.0,newInst);
				 a.setDataset(data);
				 data.add(a);
			}
		}
		if(new File("aaa.arff").exists()){
		new CommonMethords().saveInFileExsited(data, "aaa.arff");}
		else{
			new CommonMethords().saveInstances(data, "aaa.arff");
		}
	}
	public static void main(String[] args) throws Exception{
	}

}
