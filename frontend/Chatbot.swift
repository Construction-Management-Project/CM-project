//
//  Chatbot.swift
//  COREX
//
//  Created by snlcom on 5/8/25.
//

import Foundation

class Chatbot: ObservableObject {
    @Published var messages: [ChatMessage] = [] // collects the chat messages as a list
    @Published var userInput: String = "" // receiving user input as string to send to API
    
    func sendMessage() {
        let input = userInput.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !input.isEmpty else {return} // stops and returns when the input is empty
        
        let userMessage = ChatMessage(text: input, isFromUser: true) //creates a ChatMessage with the input text, setting isFromUser to true
        messages.append(userMessage) //adds the userMessage to the messages collection
        userInput = "" // empties the 'userInput' variable for next chat input
        
        /// NEED TO CONSTRUCT API CONNECTION
    }
}
