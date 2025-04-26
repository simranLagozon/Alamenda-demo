// Theme toggle functionality
function initThemeToggle() {
  const html = document.documentElement;
  const themeToggle = document.querySelector('.theme-toggle');
  const header = document.querySelector('.header');
  const navLinks = document.querySelectorAll('.nav');

  // Load saved theme or default to light
  const savedTheme = localStorage.getItem('theme') || 'light';
  html.dataset.theme = savedTheme;
  themeToggle.textContent = savedTheme === 'light' ? '🌙' : '☀️';

  themeToggle.addEventListener('click', () => {
    const newTheme = html.dataset.theme === 'light' ? 'dark' : 'light';
    html.dataset.theme = newTheme;
    localStorage.setItem('theme', newTheme);
    themeToggle.textContent = newTheme === 'light' ? '🌙' : '☀️';
    
    // Update element styles
    const textColor = newTheme === 'light' ? '#000' : '#fff';
    const headerBg = newTheme === 'light' ? 'rgba(0, 0, 0, 0.05)' : 'rgba(255, 255, 255, 0.05)';
    
    header.style.backgroundColor = headerBg;
    navLinks.forEach(link => link.style.color = textColor);
  });
}

// Initialize everything
window.addEventListener('load', () => {
  initThemeToggle();

  // Add click handlers for all buttons
  document.querySelectorAll('button').forEach(button => {
    button.addEventListener('click', (e) => {
      if (e.target.classList.contains('theme-toggle')) {
        // Theme toggle handled separately
        return;
      }
      
      if (e.target.classList.contains('demo-btn')) {
        // Handle demo button click
        console.log('Demo button clicked!');
        // Add your demo booking logic here
      }
    });
  });
});