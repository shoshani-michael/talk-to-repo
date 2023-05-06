'use client';

import { useState, useEffect, useRef } from 'react'

import Head from 'next/head'
import Header from '../components/Header'
import ChatMessages from '../components/ChatMessages'
import InputBar from '../components/InputBar'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';

export default function Home() {
    const [messages, setMessages] = useState([])
    const [collectedCodeBlocks, setCollectedCodeBlocks] = useState([]);
    const [input, setInput] = useState('')
    const inputRef = useRef(null)
    const [expandedBlocks, setExpandedBlocks] = useState(new Set());

    const handleKeyDown = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            handleSubmit(e)
        }
    }

    const handleCollectCodeBlock = (code) => {
        setCollectedCodeBlocks((prevBlocks) => [...prevBlocks, code]);
    };
      
    const clearMessages = () => {
        setMessages([]);
        setCollectedCodeBlocks([]); 
    }

    const toggleCodeBlock = (index) => {
        setExpandedBlocks((prevExpanded) => {
          const updatedExpanded = new Set(prevExpanded);
          if (prevExpanded.has(index)) {
            updatedExpanded.delete(index);
          } else {
            updatedExpanded.add(index);
          }
          return updatedExpanded;
        });
      };
      
    const getSystemMessage = async (userInputMessage) => {
        const response = await fetch('http://localhost:8000/system_message', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(userInputMessage),
        })
        const systemMessage = await response.json();
        return { text: systemMessage.system_message, sender: 'systemMessage' }
    }

    const handleSubmit = async (e) => {
        e.preventDefault()

        let updatedMessages = []
        if (input.trim()) {
            const userInputMessage = { text: input, sender: 'user' }
            if (messages.length === 0) {
                const systemMessage = await getSystemMessage(userInputMessage);
                updatedMessages = [systemMessage, userInputMessage];
            } else {
                updatedMessages = [...messages, userInputMessage]
            }
            setMessages(updatedMessages)

            await handleChat(updatedMessages)

            setInput('')
        }
    }

    useEffect(() => {
        if (inputRef.current) {
            inputRef.current.style.height = 'auto'
            inputRef.current.style.height = inputRef.current.scrollHeight + 'px'
        }
    }, [input])

    const handleChat = async (updatedMessages) => {
        let accumulatedText = "";
        fetch('http://localhost:8000/chat_stream', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(updatedMessages),
        })
        .then(response => {
            const reader = response.body.getReader();
            return new ReadableStream({
                async start(controller) {
                    while (true) {
                        const { done, value } = await reader.read();
                        if (done) {
                            break;
                        }
                        let newToken = new TextDecoder().decode(value);
                        accumulatedText += newToken;
                        controller.enqueue(newToken);
                    }
                    controller.close();
                    reader.releaseLock();
                }
            });
        })
        .then(stream => {
            updatedMessages = [...updatedMessages, { text: '', sender: 'llm' }];
            setMessages(updatedMessages);
            const reader = stream.getReader();
            reader.read().then(function processText({ done, value }) {
                if (done) {
                    return;
                }
                setMessages((prevMessages) => {
                    let outputMessage = prevMessages[prevMessages.length - 1];
                    outputMessage.text = accumulatedText;
                    return [...prevMessages.slice(0, -1), outputMessage];
                });
                return reader.read().then(processText);
            });
        });
    };

    return (<>
        <Head>
            <title>Talk to Repo</title>
            <meta
                name="description"
                content="Load any GitHub repository in a chat app."
            />
            <link rel="icon" href="/favicon.ico" />
            <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap" rel="stylesheet" />
        </Head>

        <div className="h-screen flex flex-col bg-gray-800 text-gray-100 font-sans font-roboto">
            <Header clearMessages={clearMessages} messages={messages}  />
            <div className="flex-1 overflow-auto p-4">
                <div className="flex flex-wrap md:flex-nowrap justify-center md:space-x-4">
                    <div className="w-full md:w-2/3 md:max-w-xl order-last md:order-none">
                        {/* Wrap ChatMessages and collected code blocks in a flex container */}
                        <div className="flex justify-between">
                        {/* Leave ChatMessages unchanged */}
                        <ChatMessages messages={messages} onCollectCodeBlock={handleCollectCodeBlock} />

                        {/* Add a new div for code blocks */}
                        <div className="flex-1 md:ml-4 max-w-md overflow-auto">
                            {collectedCodeBlocks.map((code, index) => (
                            <div
                                key={index}
                                className="whitespace-pre-wrap bg-gray-100 text-gray-800
                                max-w-xs text-xs p-2 rounded-lg shadow-md cursor-pointer my-2"
                                onClick={() => toggleCodeBlock(index)}
                            >
                                <SyntaxHighlighter
                                style={oneDark}
                                customStyle={{
                                    backgroundColor: "#2d2d2d",
                                    borderRadius: "0.375rem",
                                    padding: "1rem",
                                }}
                                >
                                {code}
                                </SyntaxHighlighter>
                            </div>
                            ))}
                        </div>
                        </div>
                    </div>
                </div>
            </div>

            <div className="border-t border-gray-700">
                <InputBar
                    input={input}
                    setInput={setInput}
                    handleKeyDown={handleKeyDown}
                    handleSubmit={handleSubmit}
                />
            </div>
        </div>     
    </>);    
}