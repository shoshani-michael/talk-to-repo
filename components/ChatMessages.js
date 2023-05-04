import React, { useState, useEffect } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';

const ChatMessages = ({ messages }) => {
  const [fileData, setFileData] = useState([]);

  const filteredMessages = messages.filter(
    (message) =>
      !(message.sender === 'systemMessage' | (message.sender !== 'user' && message.text.length === 0)),
  );
  
  const [showSystemMessages, setShowSystemMessages] = useState(false);

  const codeBlockRegex = /(```[\w]*[\s\S]+?```)/g;
  const copyCodeToClipboard = (code) => {
    if (navigator.clipboard) {
      navigator.clipboard.writeText(code).then(() => {
        console.log("Code copied to clipboard.");
      }).catch(err => {
        console.error("Error copying code to clipboard: ", err);
      });
    } else {
      console.error("Clipboard API not supported.");
    }
  };

  const getDisplayMessages = () => {
    if (showSystemMessages) {
      return messages;
    } else {
      return messages.filter(
        (message) =>
          !(message.sender === 'systemMessage' || (message.sender !== 'user' && message.text.length === 0))
      );
    }
  };
  
  const escapeInnerBackticks = (str) => {
    let nesting = 0;
    let escaped = '';
  
    // Split and iterate over the parts based on code block regex
    const parts = str.split(codeBlockRegex);
    parts.forEach((part, index) => {
      if (codeBlockRegex.test(part)) {
        nesting += 1;
  
        // Escape the backticks only if the current nesting level is greater than 1
        if (nesting > 1) {
          escaped += part.replace(/```/g, '&#96;&#96;&#96;');
        } else {
          escaped += part;
        }
  
        // Decrement nesting when closing backticks are encountered
        if (nesting > 0) nesting -= 1;
      } else {
        escaped += part;
      }
    });

    return escaped;
  }
  const formatMessage = (text) => {
    const parts = text.split(codeBlockRegex);
  
    return parts.map((part, index) => {
      if (codeBlockRegex.test(part)) {
        const languageRegex = /```(\w*)\n/;
        const languageMatch = part.match(languageRegex);
        const language = languageMatch && languageMatch[1] ? languageMatch[1] : '';
        const collectButton = (
            <button
                style={{
                    position: 'absolute',
                    top: 0,
                    right: 0,
                    padding: '2px',
                    borderRadius: '2px',
                    borderBottomLeftRadius: '5px',
                    fontSize: '14px',
                    background: 'rgba(255, 255, 255, 0.1)',
                    color: 'white',
                    cursor: 'pointer'
                }}
                onClick={() => {
                  copyCodeToClipboard(part.replace(languageRegex, "").replace(/```$/, ""));
                }}
            >
                collect
            </button>
        );
        return (
          <div key={index} style={{ position: 'relative' }}>
          {collectButton}
          <SyntaxHighlighter
            language={language}
            style={oneDark}
            customStyle={{ backgroundColor: '#2d2d2d', borderRadius: '0.375rem', padding: '1rem' }}
          >
            {escapeInnerBackticks(part.replace(languageRegex, '').replace(/```$/, ''))}
          </SyntaxHighlighter>
          </div>
        );
      } else {
        return (
          <span
            key={index}
            dangerouslySetInnerHTML={{
              __html: replaceFileNamesWithLinks(part, fileData),
            }}
          />
        );
      }
    });
  };

  useEffect(() => {
    async function fetchData() {
      const response = await fetch('api/fetchCsvData', {
          headers: {
            'Content-Type': 'application/json',
          },
      })
      .catch((error) => {
        console.log(error)
      });
      const data = await response.json();
      setFileData(data);
    };
    fetchData();
  }, []);

  const replaceFileNamesWithLinks = (text, data) => {
    const validExtensions = ['.md', '.py', '.scala', '.java', '.cpp', '.rs'];
    const filenameRegex = /`?((?:[\w/_-]+\/README)|[\w-_]+(?:\.(?:md|py|scala|java|cpp|rs)))`?/g;
    return text.replace(filenameRegex, (match, filename) => {
      // Check if the filename is "README" with a full file path
      const isReadmeWithPath = filename.endsWith('/README');
      const fileExtension = filename.split('.').pop();
      const isValidExtension = validExtensions.includes(`.${fileExtension}`);
  
      if (isReadmeWithPath) {
        // Use the full file path directly to create the URL for "README"
        const url = `https://github.com/twitter/the-algorithm/blob/main/${filename}`;
        return `<a class="text-blue-500 underline cursor-pointer" href="${url}" target="_blank" rel="noopener noreferrer">${filename}</a>`;
      } else if (isValidExtension) {
        // Find the record in the data list that matches the basename and has a valid extension
        const fileRecord = data.find((record) => {
          const recordBasename = record.file_name.split('/').pop();
          return recordBasename === filename;
        });
        if (fileRecord) {
          const url = `https://github.com/twitter/the-algorithm/blob/main/${fileRecord.file_name}`;
          return `<a class="text-blue-500 underline cursor-pointer" href="${url}" target="_blank" rel="noopener noreferrer">${filename}</a>`;
        }
      }
      return filename;
    });
  };

  return (
    <div className="w-full md:w-1/2 md:max-w-xl">
      {getDisplayMessages().map((message, index) => (
        <div
          key={index}
          className={`mb-4 p-3 text-lg rounded-lg shadow-md whitespace-pre-wrap ${
            message.sender === 'user' ? 'bg-gray-100 text-gray-800' : 'bg-gray-600'
          }`}
          onClick={() => {
            if (message.sender === 'user') {
              setShowSystemMessages(!showSystemMessages);
            }
          }}
        >
          {formatMessage(message.text)}
        </div>
      ))}
    </div>
  );
};

export default ChatMessages;