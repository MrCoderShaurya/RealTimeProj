import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;

public class SDT{
    public static void main(String[] args){
      Scanner sc = new Scanner(System.in);
      int [] marks  = new int[3];

      System.out.println("Enter the name of the student: ");
      String name = sc.nextLine();

      for (int i = 0; i < 3; i++){
        System.out.println("Enter marks for subject " + (i + 1) + ": ");
        marks[i] = sc.nextInt();
      }
      Student student = new Student(name, marks);
      student.printReport();

      try {
        FileWriter writer = new FileWriter("report.txt", true);
        writer.write(student.getFormattedReport());
        writer.close();
        System.out.println("Report saved to report.txt");
      } catch (IOException e) {
        System.out.println("An error occurred while saving the report.");
      }
      sc.close();
      }
    }
