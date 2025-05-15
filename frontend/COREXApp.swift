//
//  COREXApp.swift
//  COREX
//
//  Created by snlcom on 5/8/25.
//

import SwiftUI

@main
struct COREXApp: App {
    @StateObject var Bot = Chatbot()
    
    var body: some Scene {
        WindowGroup {
            MainView()
                .environmentObject(Bot)
        }
    }
}
