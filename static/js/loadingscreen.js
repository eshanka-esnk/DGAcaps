document.addEventListener('DOMContentLoaded', (event) => {
    document.getElementById("content").style.display = "initial";
    document.getElementById("loading").style.display = "none";
  });

  document.addEventListener('submit', (event) => {
    document.getElementById("loading").style.display = "flex";
    document.getElementById("content").style.display = "none";
  });