/**
 * @version
 * @function 用来处理文档分词，并输入到另一个文档里
 */
package preprocess;

import java.io.BufferedReader;
import java.io.FileNotFoundException;

import java.io.BufferedWriter;

import java.io.File;

import java.io.FileReader;

import java.io.FileWriter;

import java.io.IOException;
import java.util.HashSet;

import org.wltea.analyzer.core.IKSegmenter;
import org.wltea.analyzer.core.Lexeme;

/**
 * @author clark
 *
 */
public class Segmenter {
	
	private String sourceDir;

	private String targetDir;
	
	private HashSet<String> stopwords;

	   

	public Segmenter( String source, String target ) throws IOException {

	      sourceDir = source;

	      targetDir = target;
	      

	      stopwords = new HashSet<String>();
	     
	      //得到停词
	      FileReader fr1 = new FileReader("ex_stopword1.dic");
	      BufferedReader br1 = new BufferedReader(fr1);
	        
	      String token ="";
	        
	      while( (token = br1.readLine()) != null && token.length() != 0)
	        stopwords.add(token);

	}

	public void segment() {
		segmentDir( sourceDir, targetDir );
	}

	public void segmentDir( String source, String target ) {

	    File[] file = (new File( source )).listFiles();

	    for (int i = 0; i < file.length; i++) {

	       if (file[i].isFile()) {

	          segmentFile( file[i].getAbsolutePath(), target +File.separator + file[i].getName() );

	       }

	       if (file[i].isDirectory()) {

	          String _sourceDir = source + File.separator + file[i].getName();

	          String _targetDir = target + File.separator + file[i].getName();

	          (new File(_targetDir)).mkdirs();

	          segmentDir( _sourceDir, _targetDir );
	       }

	   }

	}

	public void segmentFile( String sourceFile, String targetFile ) {
		try {
			FileReader fr = new FileReader( sourceFile );

	        BufferedReader br = new BufferedReader(fr);

	        FileWriter fw = new FileWriter( targetFile );

	        BufferedWriter bw = new BufferedWriter(fw );

	        IKSegmenter  seg = new IKSegmenter ( br , true); 

	        Lexeme l = null;
	        


	        while( (l = seg.next()) != null){
	          //String token= l.getLexemeText();
	          //System.out.println(token);
	        	String word = l.getLexemeText();
	        	if(!stopwords.contains(word) && word.length() > 1 &&
	        			word.matches("[\u4E00-\u9FA5]+") ){
	  	          bw.write( word );
		  	      bw.write(' ');
	        	}
	        }
	        
	        bw.close();

	        fw.close();
	        

	    } catch( IOException e ) {

	        e.printStackTrace();

	    }

	 }

	public static void main( String[] args ) throws Exception {
		
	   //Segmenter segmenter = new Segmenter( "TextData", "TextData_segmented" );
	   
	   Segmenter segmenter = new Segmenter( "content", "content_segmented" );

	    segmenter.segment();
	}
}
