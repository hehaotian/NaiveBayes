import java.io.*;
import java.util.*;

public class binarize {

   public static void main(String[] args) throws IOException {
      Scanner file = new Scanner(new File(args[0]));
      PrintStream ps = new PrintStream(args[1]);
      
      String line = "";
      while (file.hasNextLine()) {
         if (file.hasNextLine()) {
            line = file.nextLine();
            line = line.replaceAll("([:])([0-9]+)", ":1");
            ps.println(line);
         }
      }
   }
   
}