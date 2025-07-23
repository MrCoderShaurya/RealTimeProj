class TodoApp{
  constructor(){
    this.taskList = document.getElementById('taskList');
    this.taskInput = document.getElementById('taskInput');
    this.tasks = JSON.parse(localStorage.getItem('tasks')) || [];
    this.renderTasks();
  }
  addTask() {
    const taskText = this.taskInput.value.trim();
    if (taskText === "")return;
    this.tasks.push(taskText);
    this.taskInput.value = "";
    this.updateLocalStorage();
    this.renderTasks();
    
  }
  deleteTask(index){
    this.tasks.splice(index,1);
    this.updateLocalStorage();
    this.renderTasks();
  }
  renderTasks(){
    this.taskList.innerHTML = "";
    this.tasks.forEach((task,index) => {
      const li = document.createElement('li');
      li.innerHTML = `${task} <button onclick="todoApp.deleteTask(${index})">Delete</button>`;
      this.taskList.appendChild(li);
    });
  }
  updateLocalStorage(){
    localStorage.setItem('tasks', JSON.stringify(this.tasks));
  }
}
const todoApp = new TodoApp();