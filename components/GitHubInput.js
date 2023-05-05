import React, { useState, useEffect } from "react";

function GitHubInput(props) {
  const [username, setUsername] = useState("");
  const [repo, setRepo] = useState("");
  const [token, setToken] = useState("");
  const [loading, setLoading] = useState(false);
  const [loadingStatus, setLoadingStatus] = useState(null);
  
  useEffect(() => {
    const storedUsername = localStorage.getItem("username");
    const storedRepo = localStorage.getItem("repo");
    const storedToken = localStorage.getItem("token");
  
    if (storedUsername) setUsername(storedUsername);
    if (storedRepo) setRepo(storedRepo);
    if (storedToken) setToken(storedToken);
  }, []);

  const Spinner = () => (
    <div className="animate-spin w-5 h-5 border-t-2 border-white border-solid rounded-full align-items-center " />
  );
  
  const handleClick = async () => {
    setLoading(true); // Start loading
    setLoadingStatus(null);
    
    try {
      const response = await fetch(process.env.NEXT_PUBLIC_BACKEND_URL + "/load_repo", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, repo, token }), 
      });

      const data = await response.json();
      console.log(data);
      
      setLoading(false); // Stop loading
      setLoadingStatus(response.ok ? 'success' : 'error');
      if (response.ok) {
        localStorage.setItem("username", username);
        localStorage.setItem("repo", repo);
        localStorage.setItem("token", token);
        props.clearMessages(); 
      }
    } catch (error) {
      console.error(error);
      setLoading(false); // Stop loading
      setLoadingStatus('error');
    }
  };

  return (
    <div className="fixed top-0 left-0 h-screen w-48 bg-gray-700 text-white p-4">
      <h2 className="text-md font-medium mb-4">Load Repository</h2>
      <input
        type="text"
        value={username}
        placeholder="GitHub Username"
        className="mb-2 w-full text-sm p-1 border rounded-lg focus:outline-none focus:ring focus:border-blue-300 resize-none overflow-hidden bg-gray-600 text-gray-100"        
        onChange={(e) => setUsername(e.target.value)}
      />
      <input
        type="text"
        value={repo}
        placeholder="Repository Name"
        className="mb-2 w-full text-sm p-1 border rounded-lg focus:outline-none focus:ring focus:border-blue-300 resize-none overflow-hidden bg-gray-600 text-gray-100"        
        onChange={(e) => setRepo(e.target.value)}
      />
      <input
        type="text"
        value={token}
        placeholder="GitHub API Token"
        className="mb-2 w-full text-sm p-1 border rounded-lg focus:outline-none focus:ring focus:border-blue-300 resize-none overflow-hidden bg-gray-600 text-gray-100"
        onChange={(e) => setToken(e.target.value)}
      />
   <button
     onClick={handleClick}
     className="w-full mb-2 text-sm px-1 py-1 rounded-lg bg-blue-500 text-white focus:outline-none hover:bg-blue-600 md:px-2 md:py-1 flex justify-center items-center"
   >
     {loading ? (
       <Spinner />
     ) : loadingStatus === 'success' ? (
       <span>✔</span>
     ) : (
       'Load Repo'
     )}
   </button>
   
    </div>
  );
}

export default GitHubInput;