/**
 * @version
 */
package preprocess;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.StringReader;

import org.wltea.analyzer.core.IKSegmenter;
import org.wltea.analyzer.core.Lexeme;
/**
 * @author clark
 *
 */
public class TestIKanalyzer {

	public static void main(String[] args) throws IOException{
		System.out.println("ok");
		String text = "我说真的";
		StringReader br = new StringReader(text);
		IKSegmenter  seg = new IKSegmenter ( br , true); 
		Lexeme l = null;
        while( (l = seg.next()) != null){
        	System.out.println(l.getLexemeText());
        }
	}
}
