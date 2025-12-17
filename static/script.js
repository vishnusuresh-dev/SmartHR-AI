
        function openChat() {
            document.getElementById('chatModal').classList.add('active');
        }

        function closeChat() {
            document.getElementById('chatModal').classList.remove('active');
        }

        // Close modal when clicking outside
        document.getElementById('chatModal').addEventListener('click', function(e) {
            if (e.target === this) {
                closeChat();
            }
        });