import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

public class Student {
  private String name;
  private int[] marks = new int [3];
  private int total;
  private double percentage;
  private String grade;
  private String timestamp;

  public Student(String name, int[] marks) {
      this.name = name;
      this.marks = marks;
      calculate();
      generateTimestamp();
  }
  private void calculate(){
    total =0;
    for (int mark : marks){
      total += mark;
    }
    percentage = total / 3.0;
    if (percentage >= 90)grade = "A+";
    else if (percentage >= 75) grade = "A";
    else if (percentage >= 60) grade = "B";
    else if (percentage >= 50) grade = "C";
    else grade = "F";
    }
  private void generateTimestamp(){
    timestamp = LocalDateTime.now()
             .format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"));
  }
  public void printReport(){
    System.out.println("\n REPORT");
    System.out.println("Name: " + name);
    System.out.println("Marks: " + marks[0] + ", " + marks[1] + ", " + marks[2]);
    System.out.println("Total: " + total);
    System.out.println("Percentage: " + percentage);
    System.out.println("Grade: " + grade);
  }
  public String getFormattedReport(){
    return "[" + timestamp + "] " + name + ": " + percentage + "% - Grade: "+ grade + "\n";
  }
}